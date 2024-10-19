import argparse
import gc
import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import LinkNeighborLoader

from config import PROJECT_ROOT
from lib.plotting import visualize_graph_plotly, create_nx_subgraph, plot_metric
from lib.poi_logger import POILog, LogLevel
from model.model_definition import CheckinScorer
from preprocessing.graph_loader import load_graph_data

tag = "Training"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

debug = {
    "Print Loaders": False,
    "Print Model": True,
}


def split_data(data, batch_size, num_neighbors, neg_sampling_ratio):
    """
    Split the graph data into training, validation, and test sets, and create data loaders for link prediction tasks
    with neighbor sampling, where negative edges are sampled dynamically during training.

    Parameters:
    -----------
    data : HeteroData
        The whole graph data to be split for link prediction.
    batch_size : int
        The size of mini-batches used by the data loaders.
    num_neighbors : list of int
        A list specifying the number of neighbors to sample at each hop during neighbor sampling.
        For example, [20, 10] means sampling 20 neighbors in the first hop and 10 neighbors in the second hop.
    neg_sampling_ratio : float
        The ratio of negative samples to positive samples during training.
        A value of 2.0 means generating two negative samples for each positive sample.

    Returns:
    --------
    train_loader : LinkNeighborLoader
        A PyTorch Geometric data loader for training, with neighbor sampling and on-the-fly negative sampling.
    val_loader : LinkNeighborLoader
        A PyTorch Geometric data loader for validation, with neighbor sampling.
    test_loader : LinkNeighborLoader
        A PyTorch Geometric data loader for testing, with neighbor sampling.

    Splitting Strategy:
    -------------------
    The function uses `RandomLinkSplit` to partition the edges into:
    - 10% for validation,
    - 10% for testing,
    - and the remaining 80% for training, with 30% disjoint training ratio.

    Negative sampling is performed during training on-the-fly,
    while validation and testing are performed using the provided edges.
    """

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=False,  # Negative samples generated on-the-fly
        is_undirected=True,
        edge_types=("user", "checkin", "venue"),
        rev_edge_types=("venue", "rev_checkin", "user"),
    )
    train_data, val_data, test_data = transform(data)

    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the `loader.LinkNeighborLoader` from PyG:

    # - - - - - - - - - - - - - - - - - TRAINING SPLIT - - - - - - - - - - - - - - - - -
    edge_label_index = train_data["user", "checkin", "venue"].edge_label_index
    edge_label = train_data["user", "checkin", "venue"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=neg_sampling_ratio,
        edge_label_index=(("user", "checkin", "venue"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    if debug["Print Loaders"]:
        sampled_data = next(iter(train_loader))
        POILog.d(tag, f"Sampled training subgraph: \n\n{sampled_data}\n")

    # - - - - - - - - - - - - - - - - - - TEST SPLIT - - - - - - - - - - - - - - - - - -
    edge_label_index = test_data["user", "checkin", "venue"].edge_label_index
    edge_label = test_data["user", "checkin", "venue"].edge_label

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors,
        edge_label_index=(("user", "checkin", "venue"), edge_label_index),
        edge_label=edge_label,
        batch_size=int((neg_sampling_ratio + 1) * batch_size),
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    if debug["Print Loaders"]:
        sampled_data = next(iter(test_loader))
        POILog.d(tag, f"Sampled testing subgraph: \n\n{sampled_data}\n")

    # - - - - - - - - - - - - - - - - - VALIDATION SPLIT - - - - - - - - - - - - - - - - -
    edge_label_index = val_data["user", "checkin", "venue"].edge_label_index
    edge_label = val_data["user", "checkin", "venue"].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neighbors,
        edge_label_index=(("user", "checkin", "venue"), edge_label_index),
        edge_label=edge_label,
        batch_size=int((neg_sampling_ratio + 1) * batch_size),
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    if debug["Print Loaders"]:
        sampled_data = next(iter(val_loader))
        POILog.d(tag, f"Sampled validation subgraph: \n\n{sampled_data}\n")

    return train_loader, val_loader, test_loader


def train(loader, model, optimizer, criterion):
    model.train()
    total_loss = total_examples = 0

    POILog.d(tag, ">> Processing batches...")

    for sampled_data in tqdm.tqdm(loader, disable=True):
        optimizer.zero_grad()
        sampled_data.to(device)

        pred = model(sampled_data)
        ground_truth = sampled_data["user", "checkin", "venue"].edge_label

        loss = criterion(pred, ground_truth)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

    epoch_loss = total_loss / total_examples
    return epoch_loss


def test(loader, model, criterion):
    model.eval()
    preds, ground_truths = [], []
    total_loss = total_examples = 0

    for sampled_data in tqdm.tqdm(loader, disable=True):
        with torch.no_grad():
            sampled_data.to(device)

            pred = model(sampled_data)
            preds.append(pred)
            ground_truth = sampled_data["user", "checkin", "venue"].edge_label
            ground_truths.append(ground_truth)

            loss = criterion(pred, ground_truth)
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

    loss = total_loss / total_examples

    # Concatena le predizioni e le ground truth
    preds = torch.cat(preds, dim=0).cpu().numpy()
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()

    out_sigmoid = torch.Tensor(preds).sigmoid().cpu().numpy()
    out_probabilities = np.rint(out_sigmoid)

    auc_sc = roc_auc_score(ground_truths, out_sigmoid)
    f1_sc = f1_score(ground_truths, out_probabilities)

    return loss, auc_sc, f1_sc


def objective(trial, data, train_loader, val_loader, epochs, patience):
    # Set of proposed values of each parameter
    hidden_channels = trial.suggest_int("hidden_channels", 16, 64)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    POILog.d(tag, f"Trial [{trial.number}]. Trying with:")
    POILog.d(tag, f" - Hidden Channels: {hidden_channels}")
    POILog.d(tag, f" - Learning Rate: {learning_rate}")
    POILog.d(tag, f" - Dropout Rate: {dropout_rate}")
    POILog.d(tag, f" - Weight Decay: {weight_decay}\n")

    model = CheckinScorer(
        num_users=data["user"].num_nodes,
        num_venues=data["venue"].num_nodes,
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Early stopping parameters
    patience_counter = 0

    best_loss = float('inf')
    best_val_auc = 0

    train_history, valid_scores, test_scores = [], [], []
    for epoch in range(epochs):
        POILog.d(tag, f"Epoch [{epoch + 1}/{epochs}]")

        # TRAINING step
        train_loss = train(train_loader, model, optimizer, criterion)
        train_history.append(train_loss)
        POILog.d(tag, f" - Train Loss: {train_loss:.4f}")

        val_loss, val_auc_score, val_f1_score = test(val_loader, model, criterion)
        valid_scores.append((val_auc_score, val_f1_score))

        POILog.d(tag, f" - Validation LOSS: {val_loss:.4f}")
        POILog.d(tag, f" - Validation AUC: {100 * val_auc_score:.2f}%")
        POILog.d(tag, f" - Validation F1: {100 * val_f1_score:.2f}%")

        # Report for Optuna
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Check for early stopping
        if val_loss < best_loss or val_auc_score > best_val_auc:
            best_loss = val_loss
            best_val_auc = val_auc_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                POILog.w(tag, ">> EARLY STOPPING TRIGGERED.\n")
                break

    return best_loss


def main(args):
    gc.collect()
    torch.cuda.empty_cache()

    POILog.MAX_LOG_LEVEL = LogLevel.DEBUG
    debug = {
        "Graph": True,
        "Graph Drawing": True,
        "Visualize Graph": False,
        "Model": True,
        "Training": True,
        "Loading": True
    }

    # - - - - - - - - - - - GRAPH DATA - - - - - - - - - - - #
    users_csv_path = f"{PROJECT_ROOT}/Dataset/users.csv"
    venues_csv_path = f"{PROJECT_ROOT}/Dataset/venues.csv"
    checkins_csv_path = f"{PROJECT_ROOT}/Dataset/checkins.csv"

    POILog.i(tag, ">> Loading the users...")
    users = pd.read_csv(users_csv_path)
    POILog.i(tag, "Users loaded.\n")

    POILog.i(tag, ">> Loading the venues...")
    venues = pd.read_csv(venues_csv_path)
    POILog.i(tag, "Venues loaded.\n")

    POILog.i(tag, ">> Loading the checkins...")
    checkins = pd.read_csv(checkins_csv_path)
    POILog.i(tag, "Checkins loaded.\n")

    POILog.i(tag, ">> Building the graph data...")
    graph_data = load_graph_data(venues, users, checkins, device, debug=False)
    POILog.i(tag, "Graph built.\n")

    if debug["Graph"]:
        POILog.d(tag, "GRAPH PROPERTIES")
        POILog.d(tag, f" - Number of nodes: {graph_data.num_nodes}")
        POILog.d(tag, f" - Number of edges: {graph_data.num_edges}")
        POILog.d(
            tag, f"Average node degree: {graph_data.num_edges / graph_data.num_nodes:.6f}"
        )
        POILog.d(tag, f" - Contains isolated nodes: {graph_data.has_isolated_nodes()}")
        POILog.d(tag, f" - Contains self-loops: {graph_data.has_self_loops()}")
        POILog.d(tag, f" - Is undirected: {graph_data.is_undirected()}")
        POILog.d(tag, f" - Graph data: \n\n{graph_data}\n")

    if debug["Visualize Graph"]:
        if debug["Graph Drawing"]: POILog.d(tag, "=== Graph visualization =================================")
        visualize_graph_plotly(create_nx_subgraph(graph_data))
        POILog.i(tag, "Network drawn.\n")

    # - - - - - - - - - - CREATE LOADERS - - - - - - - - - - - - #
    batch_size = 16
    num_neighbors = [20, 10]
    neg_sampling_ratio = 2.0

    train_loader, val_loader, test_loader = split_data(graph_data, batch_size, num_neighbors, neg_sampling_ratio)

    # - - - - - - - - - MODEL OPTIMIZATION - - - - - - - - - - - #
    model_path = f"{PROJECT_ROOT}/model/best_model.pth"
    epochs = args.epochs
    patience = 3

    already_optimized = os.path.exists(model_path)

    if not already_optimized:
        POILog.w(tag, ">> No model found.", suffix="\n\n")
        POILog.i(tag, ">> Starting the hyperparameters optimization...")
        n_trials = args.trials

        # Create the Optuna study, which will be used for hyperparameters optimization
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            study_name="To the Infinity and Beyond"
        )
        study.optimize(lambda trial: objective(trial, graph_data, train_loader, val_loader, epochs, patience),
                       n_trials=n_trials)

        print()
        POILog.i(tag, "Optimization completed.", suffix="\n\n")

        # Print the best parameters found at the end of all trials
        best_params = study.best_params
        POILog.i(tag, f"Best TRIAL is [{study.best_trial.number}]. Parameters chosen:")
        for name, value in best_params.items():
            POILog.i(tag, f" - {name}: {value}")

        POILog.d(tag, f"Best VALIDATION LOSS: {study.best_value:.4f}")
    else:
        # If the optimization has already been optimized, load the best parameters
        checkpoint = torch.load(model_path)
        best_params = checkpoint['best_params']
        POILog.w(tag, "Model already optimized. Parameters loaded:")
        for name, value in best_params.items():
            POILog.w(tag, f" - {name}: {value}")

    # - - - - - - - - - BEST MODEL TRAINING - - - - - - - - - - #
    best_model = CheckinScorer(
        num_users=graph_data["user"].num_nodes,
        num_venues=graph_data["venue"].num_nodes,
        metadata=graph_data.metadata(),
        hidden_channels=best_params['hidden_channels'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)

    print()
    POILog.i(tag, ">> Training the best model...")
    optimizer = torch.optim.Adam(
        params=best_model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Early stopping parameters
    patience_counter = 0
    best_loss = float('inf')
    best_val_auc = 0

    train_loss_history, val_loss_history, val_auc_history, val_f1_scores = [], [], [], []
    for epoch in range(epochs):
        POILog.d(tag, f"Epoch [{epoch + 1}/{epochs}]")

        # TRAINING step
        train_loss = train(train_loader, best_model, optimizer, criterion)
        train_loss_history.append(train_loss)
        POILog.d(tag, f" - Train Loss: {train_loss:.4f}")

        val_loss, val_auc_score, val_f1_score = test(val_loader, best_model, criterion)
        val_loss_history.append(val_loss)
        val_auc_history.append(val_auc_score)
        val_f1_scores.append(val_f1_score)

        POILog.d(tag, f" - Validation LOSS: {val_loss:.4f}")
        POILog.d(tag, f" - Validation AUC: {100 * val_auc_score:.2f}%")
        POILog.d(tag, f" - Validation F1: {100 * val_f1_score:.2f}%\n")

        # Check for early stopping
        if val_loss < best_loss or val_auc_score > best_val_auc:
            best_loss = val_loss
            best_val_auc = val_auc_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                POILog.w(tag, ">> EARLY STOPPING TRIGGERED.\n")
                break

    if not already_optimized:
        # Save the best model to disk, including the best parameters found during optimization
        try:
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'best_params': {
                    'hidden_channels': best_params["hidden_channels"],
                    'learning_rate': best_params["learning_rate"],
                    'dropout_rate': best_params["dropout_rate"],
                    'weight_decay': best_params["weight_decay"]
                },
            }, model_path)
            POILog.w(tag, f"Model saved to '{os.path.abspath(model_path)}'.")
        except Exception as e:
            POILog.e(tag, f"Error saving model: {e}")
    else:
        POILog.w(tag, f"Model already saved to '{os.path.abspath(model_path)}'.")

    # - - - - - - - - - - - - - TEST PHASE - - - - - - - - - - - - #
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_auc_score, test_f1_score = test(test_loader, best_model, criterion)
    POILog.i(tag, "Results over the test set:")
    POILog.i(tag, f" - Test Loss: {test_loss:.4f}")
    POILog.i(tag, f" - Test AUC: {100 * test_auc_score:.2f}%")
    POILog.i(tag, f" - Test F1: {100 * test_f1_score:.2f}%")

    plot_metric({
        "Train Loss": train_loss_history,
        "Validation Loss": val_loss_history
    }, metric_name="Binary Cross-Entropy Loss", colors=['blue', 'red'], save_img=True)

    plot_metric({"Validation AUC": val_auc_history}, metric_name="AUC", colors=['orange'], save_img=True)

    plot_metric({"Validation F1 Score": val_f1_scores}, metric_name="F1-score", colors=['green'], save_img=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of epochs to train the model')
    parser.add_argument('-t', '--trials', type=int, default=25,
                        help='Number of trials for Optuna optimization')
    args = parser.parse_args()

    main(args)
