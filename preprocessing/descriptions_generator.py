from config import PROJECT_ROOT
import os
import random
import pandas as pd
import torch
import clip
from PIL import Image
from lib.poi_logger import POILog, LogLevel
from transformers import BlipProcessor, BlipForConditionalGeneration, PreTrainedModel

tag = "Desc Generator"


def generate_candidates(image, prompt: str, processor: BlipProcessor, model: PreTrainedModel, num_candidates=5, max_length=30):
    """
    Employs a BLIP model to generate a certain number of description candidates for the given image, based on the prompt.

    Parameters:
    -----------
    image : PIL.Image
        The input image to be used for generating the description candidates.
    prompt : str
        A string containing the start of the description.
    processor : BlipProcessor
        Object that preprocesses the image before feeding it to the BLIP model.
    model : PreTrainedModel
        The pre-trained BLIP model used to generate the description candidates.
    num_candidates : int
        The number of description candidates to generate.
    max_length : int
        The maximum length of the description candidates.

    Returns:
    --------
    captions : list[str]
        The description candidates generated by the BLIP model.
    """

    inputs = processor(image, text=prompt, return_tensors="pt").to("cuda")

    POILog.d(tag, " - Generating caption candidates...")

    out = model.generate(
        **inputs,
        num_return_sequences=num_candidates,
        max_length=max_length,
        num_beams=5,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.5,
    )

    captions = processor.batch_decode(out, skip_special_tokens=True)
    captions = [caption.replace("\n", " ") for caption in captions]

    for candidate_idx, caption in enumerate(captions, 1):
        POILog.d(tag, f"   [{candidate_idx}] {caption}")

    return captions


def select_best_caption(image, captions, model, preprocessor, device):
    """
    Select the best caption for a given image based on similarity scores using a CLIP model.

    Parameters:
    -----------
    image : PIL.Image
        The input image for which the best caption is to be selected.
    captions : list of str
        A list of caption candidates to be evaluated for the image.
    model : CLIP
        The pre-trained CLIP model used to encode both the image and text features.
    preprocessor : CLIPProcessor
        Object that preprocesses the image before feeding it into the model.
    device : device
        The device (CPU or GPU) on which the computation should be performed.

    Returns:
    --------
    best_caption : str
        The caption from the `captions` list that has the highest similarity to the image.
    similarity_score : float
        The similarity score of the best caption, normalized using a softmax function.
    """
    with torch.no_grad():
        processed_image = preprocessor(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(captions).to(device)

        # image_features, text_features = model(processed_image, text_inputs)

        image_features = model.encode_image(processed_image)
        text_features = model.encode_text(text_inputs)

    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    best_caption_idx = similarities.argmax().item()
    best_caption = captions[best_caption_idx]

    return best_caption, similarities[0][best_caption_idx].item()


def generate_description(image_name, blip_processor, blip_model, clip_preprocessor, clip_model, venues, device):
    """
    Employs a BLIP model to generate a description candidates for the given venue image, and chooses the most
    fitting using a CLIP model.

    Parameters:
    -----------
    image_name : str
        The file name of the input image to be used for generating the description candidates.
    blip_processor : BlipProcessor
        Object that preprocesses the image before feeding it to the BLIP model.
    blip_model : PreTrainedModel
        The pre-trained BLIP model used to generate the description candidates.
    clip_preprocessor : CLIPProcessor
        Object that preprocesses the image before feeding it into the model.
    clip_model : CLIP
        The pre-trained CLIP model used to encode both the image and text features.
    venues : dict[int, str]
        A dictionary containing the (venue_id, venue_category) pairs used to generate the prompts
    Returns:
    --------
    best_caption : str
        The generated description of the image.
    score : float
        The similarity score of the description w.r.t. the image.
    """
    venue_id = image_name.removesuffix(".jpg")

    POILog.d(tag, f" - Loading image...")
    venue_image = Image.open(f"{PROJECT_ROOT}/images/{image_name}").convert("RGB")

    venue_cat = venues[venue_id]
    venue_prompt = f"This venue is a {venue_cat}, "

    POILog.i(tag, f" - Prompt: {venue_prompt}")
    generated_captions = generate_candidates(
        venue_image,
        prompt=venue_prompt,
        processor=blip_processor,
        model=blip_model,
        num_candidates=5,
        max_length=50,
    )

    POILog.d(tag, f" - Selecting the best candidate...")
    best_caption, best_score = select_best_caption(
        venue_image,
        generated_captions,
        model=clip_model,
        preprocessor=clip_preprocessor,
        device=device,
    )

    POILog.i(tag, f"Generated caption: {best_caption} [Score: {best_score}]\n")

    return best_caption, best_score


def main():
    POILog.MAX_LOG_LEVEL = LogLevel.VERBOSE
    limit = -1
    test = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    POILog.i(tag, f"Selected device: {device}")

    POILog.i(tag, f"Loading the BLIP Processor...")
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        clean_up_tokenization_spaces=False
    )

    POILog.i(tag, f"Loading the BLIP Model...")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    POILog.i(tag, f"Loading the CLIP Model and Preprocessor...")
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)

    POILog.i(tag, "Loading the venues...")
    venues = pd.read_csv(f"{PROJECT_ROOT}/Dataset/venues.csv")
    venues_cat = {venue_id: category for venue_id, category in venues[["Venue_ID", "Venue_category"]].values}

    print()

    image_names = os.listdir(f"{PROJECT_ROOT}/images")
    venues_descriptions = []

    if test:
        image_name: str = random.choice(image_names)
        generate_description(image_name, blip_processor, blip_model, clip_preprocessor, clip_model, venues_cat, device)
        exit(0)

    num_captions = limit if limit != -1 else len(image_names)

    for idx, image_name in enumerate(image_names[:limit]):
        venue_id = image_name.removesuffix(".jpg")
        POILog.i(tag, f"[{idx + 1}/{num_captions}] Generating caption for venue {venue_id}")
        description, score = generate_description(image_name, blip_processor, blip_model, clip_preprocessor, clip_model, venues_cat, device)
        venues_descriptions.append({"Venue_ID": venue_id, "Venue_description": description})

    print()
    POILog.i(tag, f"Saving the captions...")
    pd.DataFrame(venues_descriptions).to_csv(f"{PROJECT_ROOT}/additional_data/venues_desc.csv", index=False)
    POILog.i(tag, f"Captions saved.")


if __name__ == "__main__":
    main()