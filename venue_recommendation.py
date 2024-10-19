from config import PROJECT_ROOT
import numpy as np
import os
import pandas as pd
import torch
from lib.input_utils import validate_input
from lib.plotting import visualize_graph_plotly, create_nx_subgraph
from lib.poi_logger import POILog, LogLevel
from model.model_definition import CheckinScorer
from model.recommender import evaluate_total_scores, evaluate_descriptions_scores, evaluate_user_scores
from preprocessing.description_encoder import encode_description
from preprocessing.graph_loader import load_graph_data, extract_recommendations_subgraph, extract_random_subgraph, \
    get_venues_desc_embeddings

tag = "VenueRecommendation"
POILog.MAX_LOG_LEVEL = LogLevel.VERBOSE

debug = {
    "Graph": True,
    "Graph Drawing": True,
    "Visualize Data Graph": True,
    "Recommendations subgraph": False,
    "Model": False,
    "Non-Top Recommendations": False,
}

debug_val = {
    "top_k": 50,
    "center_k": 10,
    "bottom_k": 10,
    "precision": 6,
}

# %% - - - - - - - - Env Initialization
users_csv_path = f"{PROJECT_ROOT}/Dataset/users.csv"
venues_csv_path = f"{PROJECT_ROOT}/Dataset/venues.csv"
checkins_csv_path = f"{PROJECT_ROOT}/Dataset/checkins.csv"
venue_cat_csv_path = f"{PROJECT_ROOT}/Dataset/venue_categories.csv"
descriptions_csv_path = f"{PROJECT_ROOT}/additional_data/venues_desc_mapped.csv"

it_start = "\033[3m"
it_end = "\033[0m"

desc_found = os.path.exists(descriptions_csv_path)

if not os.path.exists(users_csv_path) or not os.path.exists(venues_csv_path) or not os.path.exists(
        checkins_csv_path) or not os.path.exists(venue_cat_csv_path):
    POILog.e(tag, msg=f"At least one of the dataset files ({it_start}users.csv{it_end}, "
                      f"{it_start}venues.csv{it_end}, {it_start}checkins.csv{it_end}) does not exist.\n"
                      f"Please check the paths of the dataset or execute the {it_start}write_csv{it_end} script.")
    exit(1)

if not os.path.exists(descriptions_csv_path):
    POILog.w(tag, msg="The description file was not found!\n"
                      "Please download the file from the provided link or the description-based recommendation will not be available.\n\n")

model_path = './model/best_model.pth'

if not os.path.exists(model_path):
    POILog.e(tag, msg="The model file does not exist!\n"
                      f"Please make sure that the model file is in the correct path or execute the {it_start}model_training{it_end} script.")
    exit(1)

POILog.i(tag, "---------- Venue Recommendation System ----------")

# %% - - - - - - - - Device Detection

if torch.cuda.is_available():
    device_name = "cuda"
else:
    device_name = "cpu"

device = torch.device(device_name)

POILog.i(tag, f"Device: {device}\n")

# %% - - - - - - - - Dataset Loading
POILog.i(tag, "Model and dataset loading\n")

POILog.i(tag, ">> Loading the users...")
users = pd.read_csv(users_csv_path)

POILog.i(tag, ">> Loading the venues...")
venues = pd.read_csv(venues_csv_path)

POILog.i(tag, ">> Loading the checkins...")
checkins = pd.read_csv(checkins_csv_path)

desc_map = dict()
if desc_found:
    POILog.i(tag, ">> Loading the venues text descriptions...", suffix="\n\n")
    desc_df = pd.read_csv(descriptions_csv_path)
    desc_map = {venue_id: desc for venue_id, desc in desc_df[["Venue_ID", "Venue_description"]].to_numpy()}

POILog.i(tag, ">> Building the graph data...")
data = load_graph_data(venues, users, checkins, device, debug=False)
POILog.i(tag, "Graph data created.", suffix="\n\n")

if debug["Graph"]: POILog.d(tag, f"Graph data: \n\n{data}\n")

# %% - - - - - - - - Data Random Subgraph Visualization
if debug["Visualize Data Graph"]:
    if debug["Graph Drawing"]: POILog.d(tag, "=== Graph visualization =================================")

    data_random_subgraph = extract_random_subgraph(data, num_displayed_users=30, debug=debug["Graph Drawing"])

    if debug["Graph Drawing"]: POILog.d(tag, f"Data random subgraph: \n\n{data_random_subgraph}\n")

    nx_data_subgraph = create_nx_subgraph(data_random_subgraph)

    visualize_graph_plotly(nx_data_subgraph)

    POILog.i(tag, "Network drawn.\n")

# %% - - - - - - - - Model Loading
POILog.i(tag, ">> Loading the venue-to-user score evaluator...")
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

params = checkpoint['best_params']
model = CheckinScorer(
    num_users=data["user"].num_nodes,
    num_venues=data["venue"].num_nodes,
    metadata=data.metadata(),
    hidden_channels=params['hidden_channels'],
    dropout_rate=params['dropout_rate']
)
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
if debug["Model"]: POILog.d(tag, f"Model loaded: \n\n{model}\n")

POILog.i(tag, "Loading completed.\n")

# %% - - - - - - - - UserID Input
POILog.i(tag, ">> Choosing the User ID to generate recommendations...")

user_ids = users["User_ID"].to_numpy()

# Obtain 3 random user IDs from the dataset with more than 30 checkins
user_checkin_counts = checkins["User_ID"].value_counts()
user_checkin_counts = user_checkin_counts[user_checkin_counts > 30].index.to_numpy()
suggested_user_ids = np.random.choice(user_checkin_counts, 3, replace=False)

user_id = None
user_node_id = None
valid_id = False
while not valid_id:
    try:
        suggestion_text = ", ".join([str(_id) for _id in suggested_user_ids])
        input_prompt = f"Please enter the ID of the user (e.g. ({suggestion_text}): "
        user_id = int(validate_input(input_prompt, type=int))
        if user_id in user_ids:
            valid_id = True
            user_node_id = users[users['User_ID'] == user_id].index[0]

            POILog.i(tag, f"User [{user_id}] selected (Node ID: {user_node_id}).", suffix="\n\n")
        else:
            POILog.w(tag, "The inserted User ID was not found. Please try again.")
    except ValueError:
        POILog.w(tag, "Invalid input. Please enter a valid integer User ID.")

# %% - - - - - - - - Description Input
POILog.i(tag, ">> Requesting the description of the desired venue...")
query = validate_input("Please enter the description of the desired venue: ", max_length=150)
print()
query_embedding = encode_description(description=query, debug=True)

# %% - - - - - - - - Recommendations Generation
POILog.i(tag, ">> Recommendation list calculation")

POILog.d(tag, "Evaluating the user's compatibility with the venues...")
user_scores = evaluate_user_scores(model, data, user_node_id)

POILog.d(tag, "Evaluating the description's compatibility with the venues...")
venues_desc_embeddings = get_venues_desc_embeddings(venues)
description_scores = evaluate_descriptions_scores(query_embedding, venues_desc_embeddings, device=device)

POILog.d(tag, "Evaluating the overall compatibility...")
recommendations = evaluate_total_scores(user_scores, description_scores, user_weight=0.4, desc_weight=0.6)
POILog.i(tag, "Recommendations computed.", suffix="\n\n")

# %% - - - - - - - - Recommendations Visualization
top_k = debug_val["top_k"]
center_k = debug_val["center_k"]
bottom_k = debug_val["bottom_k"]
p = debug_val["precision"]
recommendations_list = None
top_k_recommendations = None
selected_list = "no list selected"

while True:
    print()
    POILog.i(tag, "<! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !>")
    POILog.i(tag, f"Choose an option to display the first {top_k} recommendations:")
    POILog.i(tag, "1) Print recommendations based on the user's preferences.")
    POILog.i(tag, "2) Print the recommendations based on the description entered.")
    POILog.i(tag, "3) Obtain the recommendations based on the overall compatibility.")
    POILog.i(tag, f"4) Draw the selected recommendations graph ({selected_list}).")
    POILog.i(tag, "5) Exit from the visualization.", suffix="\n\n")

    try:
        choice = int(validate_input("Enter your choice (1-5): ", type=int))
    except ValueError:
        POILog.w(tag, "Invalid input. Please enter a valid integer.")
        continue

    if choice == 1:
        selected_list = "user scores"
        recommendations_list = list(user_scores.items())
        POILog.i(tag, f"Recommendations for User [{user_id}] based on the user's preferences:", suffix="\n")
    elif choice == 2:
        selected_list = "description scores"
        recommendations_list = list(description_scores.items())
        POILog.i(tag, f"Recommendations for User [{user_id}] based on the description:", suffix="\n")
    elif choice == 3:
        selected_list = "total scores"
        recommendations_list = list(recommendations.items())
        POILog.i(tag, f"Recommendations for User [{user_id}]:", suffix="\n")
    elif choice == 4:
        if top_k_recommendations is None:
            POILog.w(tag, "No recommendations selected. Please choose a list first.")
            continue

        POILog.i(tag, ">> Recommendations Subgraph Extraction\n")

        top_k_venues_ids = [recommendation[0] for recommendation in top_k_recommendations]
        recs_subgraph = extract_recommendations_subgraph(data, user_node_id, top_k_venues_ids)

        if debug["Recommendations subgraph"]:
            POILog.d(tag, f"Recommendations subgraph: \n\n{recs_subgraph}\n")
            POILog.d(tag, "Converting the PyG subgraph to a networkx graph...")

        nx_recs_subgraph = create_nx_subgraph(recs_subgraph)

        edge_scores = {(0, idx): score for idx, (_, score) in enumerate(top_k_recommendations)}

        if debug["Recommendations subgraph"]: POILog.d(tag, "Visualizing the networkx graph...")

        visualize_graph_plotly(nx_recs_subgraph, edge_scores=edge_scores)

        POILog.i(tag, f"Network of recommendations for user [{user_id}] drawn.\n")

        continue
    elif choice == 5:
        POILog.i(tag, "Exiting...")
        break
    else:
        POILog.w(tag, "Invalid choice. Please enter a number between 1 and 4.")
        continue

    # Print the top-k recommendations
    top_k_recommendations = recommendations_list[:top_k]
    for idx, (venue_id, score) in enumerate(top_k_recommendations):
        desc_text = "No description found."

        if desc_found:
            assert desc_map is not None
            desc_text = desc_map[venue_id]

        if choice == 3:
            details = f"User Score: {user_scores[venue_id]:.{p}f} - Description Score: {description_scores[venue_id]:.{p}f} - Total"
        else:
            details = ""
        POILog.i(tag, msg=f"  {idx + 1}) Venue {venue_id} | {details} Score: {score:.{p}f}\n"
                          f"     {it_start}Description{it_end}: {desc_text}")

    if debug["Non-Top Recommendations"]:
        center_idx = round(len(recommendations) / 2.0)
        delta_idx = round(center_k / 2.0)
        center_start = center_idx - delta_idx
        center_end = center_idx + delta_idx
        center_k_recommendations = recommendations_list[center_start:center_end]

        print()

        POILog.i(tag, f"Central ~{center_k} recommendations:")
        for idx, (venue_id, score) in enumerate(center_k_recommendations):
            desc_text = "No description found."

            if desc_found:
                assert desc_map is not None
                desc_text = desc_map[venue_id]

            if choice == 3:
                details = f"User Score: {user_scores[venue_id]:.{p}f} - Description Score: {description_scores[venue_id]:.{p}f} - Total"
            else:
                details = ""
            POILog.d(tag, msg=f"  {center_start + idx + 1}) Venue {venue_id} | {details} Score: {score:.{p}f}\n"
                              f"     {it_start}Description{it_end}: {desc_text}")

        print()

        bottom_start = len(recommendations) - bottom_k
        bottom_k_recommendations = recommendations_list[bottom_start:]

        POILog.i(tag, f"Last {bottom_k} recommendations:")
        for idx, (venue_id, score) in enumerate(bottom_k_recommendations):
            desc_text = "No description found."

            if desc_found:
                assert desc_map is not None
                desc_text = desc_map[venue_id]

            if choice == 3:
                details = f"User Score: {user_scores[venue_id]:.{p}f} - Description Score: {description_scores[venue_id]:.{p}f} - Total"
            else:
                details = ""
            POILog.d(tag, msg=f"  {center_start + idx + 1}) Venue {venue_id} | {details} Score: {score:.{p}f}\n"
                              f"     {it_start}Description{it_end}: {desc_text}")
