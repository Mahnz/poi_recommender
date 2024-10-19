from config import PROJECT_ROOT
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from lib.poi_logger import POILog


tag = "DataLoader"


def extract_random_subgraph(
        data: HeteroData,
        num_displayed_users,
        num_displayed_venues=None,
        only_user_venues=True,
        debug: bool = False
):
    """
    Extracts a subgraph with the specified characteristics from a HeteroData object.

    Parameters:
    -----------
    data : HeteroData
        The original graph from which we need to extract the subgraph.
    num_displayed_users : int
        The number of users to be randomly extracted from the graph.
    num_displayed_venues : int | None
        The number of venues to be randomly extracted from the graph. If unspecified, all the venues are extracted.
    only_user_venues : bool
        Indicates whether the venues have to be extracted only among the venues that the extracted users have
        visited, or between all the venues.
    debug : bool
        If set, debug logs are displayed.

    Returns:
    --------
    subgraph : HeteroData
        The extracted subgraph.
    """
    tot_users = data["user"].num_nodes
    tot_venues = data["venue"].num_nodes

    if debug: POILog.d(tag, f"Total number of users: {tot_users}")

    user_indices = torch.randperm(tot_users)[:num_displayed_users]

    if only_user_venues:
        checkins = data["user", "checkin", "venue"].edge_index
        checkins = checkins[:, np.isin(checkins[0].cpu().numpy(), user_indices.numpy())]
        venue_indices = checkins[1].unique().cpu()
    else:
        venue_indices = torch.range(0, tot_venues, dtype=torch.int64)

    if num_displayed_venues is not None:
        venue_indices = venue_indices[torch.randperm(len(venue_indices))[:num_displayed_venues]]

    if debug:
        max_user_idx = min(10, len(user_indices))
        max_venue_idx = min(10, len(venue_indices))
        POILog.v(tag, f"Indexes of selected users (first {max_user_idx}): \n{user_indices[:max_user_idx]}\n")
        POILog.v(tag, f"Indexes of selected venues (first {max_venue_idx}): \n{venue_indices[:max_venue_idx]}\n")

    subgraph = data.clone().cpu().subgraph({
        "user": user_indices,
        "venue": venue_indices
    })

    # This is to let all the nodes have the same properties, else networkx creates problems
    subgraph["user"].x = torch.tensor([0] * num_displayed_users)

    if debug: POILog.d(tag, "Subgraph created.")

    return subgraph


def extract_recommendations_subgraph(
        data: HeteroData,
        user_node_id: int,
        venue_indices: list[int],
):
    """
    Creates a subgraph with the specified recommended venues of a certain user, from the whole graph HeteroData object.

    Parameters:
    -----------
    data : HeteroData
        The whole graph data.
    user_node_id : int
        The node ID of the user for whom the subgraph is being created.
    venue_indices : list of int
        A list of indices representing the recommended venues to include in the subgraph.

    Returns:
    --------
    subgraph : HeteroData
        The extracted subgraph of the recommendations.
    """
    # Create a subgraph with the specified user and venue nodes
    subgraph = HeteroData()

    user_index = torch.tensor([user_node_id]).cpu()
    venue_indices_tensor = torch.tensor(venue_indices).cpu()
    num_venues = len(venue_indices)

    # Add user node
    subgraph["user"].node_id = user_index

    # This is to let all the nodes have the 'x' property, else networkx creates problems
    subgraph["user"].x = torch.tensor([0])

    # Add venue nodes
    subgraph["venue"].node_id = torch.arange(num_venues)
    subgraph["venue"].x = data["venue"].x[venue_indices_tensor]

    # Create edge index for the subgraph
    user_index_repeated = torch.zeros(num_venues, dtype=torch.long)
    new_venue_indices = torch.arange(num_venues, dtype=torch.long)
    edge_index = torch.stack((user_index_repeated, new_venue_indices), dim=0)

    subgraph["user", "checkin", "venue"].edge_index = edge_index
    subgraph["user", "checkin", "venue"].edge_label = torch.ones((num_venues, 1))

    return subgraph


def load_graph_data(venues_df, users_df, checkins_df, device, debug=False):
    """
    Load and process graph data from dataframes for a heterogeneous graph of users, venues, and checkins.

    Parameters:
    -----------
    venues_df : pd.DataFrame
        DataFrame containing venue information. Must include columns for "Venue_category" and "Venue_desc_emb".
    users_df : pd.DataFrame
        DataFrame containing user information. Must include a "User_ID" column.
    checkins_df : pd.DataFrame
        DataFrame containing checkin information. Must include "User_ID", "Venue_ID", and "Recency_Score" columns.
    device : torch.device
        The device (CPU or GPU) to which the resulting graph data should be moved.
    debug : bool, optional (default=False)
        If True, debug messages are logged during processing.

    Returns:
    --------
    data : HeteroData
        A heterogeneous graph in `HeteroData` format with the following structure:
        - User nodes: containing node IDs.
        - Venue nodes: containing node IDs and features, where features are a combination of venue category and description embeddings.
        - Edges: directed edges from users to venues, representing checkins. Each edge has a recency score as an attribute.

    Example usage:
    --------------
    >>> venues_df = pd.read_csv('venues.csv')
    >>> users_df = pd.read_csv('users.csv')
    >>> checkins_df = pd.read_csv('checkins.csv')
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> graph_data = load_graph_data(venues_df, users_df, checkins_df, device, debug=True)
    >>> print(graph_data)
    """
    if debug: POILog.d(tag, ">> Processing venues...")

    categories_df = pd.read_csv(f"{PROJECT_ROOT}/Dataset/venue_categories.csv")
    category_mapping = categories_df.set_index('Venue_category')['Cat_code'].to_dict()
    venues_df['Venue_category'] = venues_df['Venue_category'].map(category_mapping)

    if debug: POILog.d(tag, f"Processed venues: \n\n{venues_df.describe()}\n")

    if debug: POILog.d(tag, f">> Processing checkins...")

    checkins = checkins_df.__copy__()
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(users_df['User_ID'])}
    checkins["User_ID"] = checkins_df["User_ID"].map(user_id_mapping)
    if debug: POILog.d(tag, f"Processed checkins: \n\n{checkins.describe()}\n")

    if debug: POILog.d(tag, ">> Creating the graph data...")

    venues_features = venues_df[["Venue_category", "Venue_desc_emb"]].values
    checkins_index = checkins[["User_ID", "Venue_ID"]].values.T

    venues_node_features = []

    for idx, (venue_category, venue_desc_emb) in enumerate(venues_features):
        if debug and (idx == 0 or (idx + 1) % 3000 == 0 or (idx + 1) == len(venues_features)):
            POILog.d(tag, f" - Analyzing venue {idx + 1}/{len(venues_features)}")
        venue_node_features = [venue_category]
        for desc_emb_el in venue_desc_emb.split(","):
            venue_node_features.append(float(desc_emb_el))

        venues_node_features.append(np.array(venue_node_features))

    venues_node_features = np.array(venues_node_features)

    data = HeteroData()

    data["user"].node_id = torch.arange(users_df.shape[0], device=device, dtype=torch.int32)

    data["venue"].node_id = torch.arange(venues_df.shape[0], device=device, dtype=torch.int32)
    data["venue"].x = torch.tensor(venues_node_features, device=device, dtype=torch.float32)

    data["user", "checkin", "venue"].edge_index = torch.tensor(
        checkins_index, device=device, dtype=torch.int64
    )

    recency_scores = checkins_df["Recency_Score"].values
    data["user", "checkin", "venue"].edge_attr = torch.tensor(
        recency_scores, device=device, dtype=torch.float32
    ).unsqueeze(1)

    data = T.ToUndirected()(data)
    data.to(device)

    return data


def emb_str_to_arr(emb_str):
    return np.array([float(emb_cell_str) for emb_cell_str in emb_str.split(",")])


def get_venues_desc_embeddings(venues_df):
    venues_emb_df = venues_df[["Venue_ID", "Venue_desc_emb"]]
    venues_emb = {venue_id: emb_str_to_arr(emb_str) for venue_id, emb_str in venues_emb_df.to_numpy()}
    return venues_emb
