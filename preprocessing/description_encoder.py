from config import PROJECT_ROOT
import pandas as pd
from lib.poi_logger import POILog
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentence_transformers import SentenceTransformer

tag = "DescEncoder"


def encode_description(description: str, debug=False):
    """
    Takes a single description text and encodes it in an embedding array of 384 float values using a
    SentenceTransformer model.
    :param description: The description to be encoded
    :param debug: Flag to indicate wether to print the debug messages regarding the loading and usage of the model
    :return: The resulting embedding as a 384-long float array.
    """
    POILog.d(tag, "Loading the description encoder...")
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    POILog.d(tag, "Model loaded.")

    if debug: POILog.d(tag, f"Encoding description: {description}")
    embedding = transformer.encode(description)
    if debug: POILog.d(tag, f"Embedding structure:  {len(embedding)} elements  [{type(embedding[0])}]\n")

    return embedding


def __embedding_to_string(embedding):
    return ','.join(map(str, embedding))


def main():
    """
    This script takes all the description texts in "venues_desc.csv" and encodes them using a SentenceTransformers model.
    The model returns an array of 384 float values, which are then converted to a comma-separated string representation.
    The resulting strings are stored in "venues_emb.csv".
    """

    descriptions_path = f"{PROJECT_ROOT}/additional_data/venues_desc.csv"
    venue_embeddings_path = f"{PROJECT_ROOT}/additional_data/venues_emb.csv"

    POILog.i(tag, "Loading the descriptions...")
    descriptions = pd.read_csv(descriptions_path)
    descriptions_data = descriptions[["Venue_ID", "Venue_description"]].values
    n_desc = len(descriptions)
    POILog.i(tag, f"{n_desc} descriptions loaded")

    POILog.i(tag, "Loading the description encoder...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    POILog.i(tag, "Model loaded.")

    embeddings = []

    for idx, (venue_id, description) in enumerate(descriptions_data):
        debug = (idx == 0) or ((idx + 1) % 100 == 0) or (idx == n_desc)

        if debug: POILog.d(tag, f"[{idx + 1}/{n_desc}] Encoding description: {description}")

        embedding = model.encode(description)

        if debug: POILog.d(tag, f"Embedding structure:  {len(embedding)} elements  [{type(embedding[0])}]")

        embedding = __embedding_to_string(embedding)

        if debug: POILog.d(tag, f"Description embedding string: {embedding[:70]}")
        if debug: print()

        embeddings.append({"Venue_ID": venue_id, "Venue_embedding": embedding})

    POILog.i(tag, "Saving embeddings...")
    pd.DataFrame(embeddings).to_csv(venue_embeddings_path, index=False)
    POILog.i(tag, "Embeddings saved.")


if __name__ == "__main__":
    main()
