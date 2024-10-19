from config import PROJECT_ROOT
import pandas as pd
from lib.poi_logger import POILog
import torch
from model.model_definition import CheckinScorer
from model.model_training import split_data
from preprocessing.graph_loader import load_graph_data

"""
The purpose of this script is testing some properties of the processed data obtained during the computations.
The tests are enabled in the ``tests`` dictionary and can be controlled by the subsequent properties.
"""

tag = "Model Test"
device_name = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device(device_name)
POILog.i(tag, f"Device: {device}")


model_path = f"{PROJECT_ROOT}/model/best_model.pth"

users_csv_path = f"{PROJECT_ROOT}/Dataset/users.csv"
venues_csv_path = f"{PROJECT_ROOT}/Dataset/venues.csv"
checkins_csv_path = f"{PROJECT_ROOT}/Dataset/checkins.csv"

venues_emb_path = f"{PROJECT_ROOT}/additional_data/venues_emb.csv"
venues_fin_path = f"{PROJECT_ROOT}/additional_data/venues_final.csv"

tests = {
    "check_venues": True,
    "check_samples": True,
    "check_model": True,
}

graph_data = None
n_samples = 20

POILog.i(tag, ">> Loading the users...")
users_df = pd.read_csv(users_csv_path)
POILog.i(tag, "Users loaded.\n")

POILog.i(tag, ">> Loading the venues...")
venues_df = pd.read_csv(venues_csv_path)
POILog.i(tag, "Venues loaded.\n")

POILog.i(tag, ">> Loading the checkins...")
checkins_df = pd.read_csv(checkins_csv_path)
POILog.i(tag, "Checkins loaded.\n")

if tests["check_venues"]:
    POILog.i(tag, ">> Loading the embeddings...")
    emb_df = pd.read_csv(venues_emb_path)
    POILog.i(tag, "Venues loaded.\n")

    POILog.i(tag, ">> Loading the final venues...")
    final_df = pd.read_csv(venues_fin_path)
    POILog.i(tag, "Venues loaded.\n")

    n_venues = len(venues_df)
    n_final = len(final_df)
    n_emb = len(emb_df)

    POILog.i(tag, f"N. venues: {n_venues}")
    POILog.i(tag, f"N. embeddings: {n_emb}")
    POILog.i(tag, f"N. final: {n_final}")
    POILog.i(tag, f"N. None: {n_venues - n_final}")

if tests["check_samples"]:
    if graph_data is None:
        POILog.i(tag, ">> Building the graph data...")
        graph_data = load_graph_data(venues_df, users_df, checkins_df, device, debug=False)
        POILog.i(tag, "Graph built.\n")

    train_loader, val_loader, test_loader = split_data(graph_data, 128, [20, 10], 2.0)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    for _ in range(n_samples):
        train_sample = next(train_iter)
        val_sample = next(val_iter)

        if train_sample is None and val_sample is None:
            break

        if train_sample is not None:
            POILog.i(tag, f"Train sample: \n\n{train_sample}\n")

        if val_sample is not None:
            POILog.i(tag, f"Validation sample: \n\n{val_sample}\n", suffix="\n\n")

    exit(0)

if tests["check_model"]:
    if graph_data is None:
        POILog.i(tag, ">> Building the graph data...")
        graph_data = load_graph_data(venues_df, users_df, checkins_df, device, debug=False)
        POILog.i(tag, "Graph built.\n")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = CheckinScorer(
        num_users=graph_data["user"].num_nodes,
        num_venues=graph_data["venue"].num_nodes,
        metadata=graph_data.metadata(),
        hidden_channels=checkpoint["best_params"]["hidden_channels"],
        dropout_rate=checkpoint["best_params"]["dropout_rate"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    POILog.i(tag, f"Model: \n{model}")
