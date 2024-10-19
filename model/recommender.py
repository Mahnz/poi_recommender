import torch
from torch import Tensor
from torch import device
from torch_geometric.data import HeteroData
from lib.poi_logger import POILog
from model.model_definition import CheckinScorer, LinkPredictor

tag = "Recommender"

debug = {
    "Recap": True,
    "Last recommendation": False,
}


def evaluate_user_scores(model: CheckinScorer, data, user_node_id, recap=False):
    """
    Computes the user-venue similarity score for each venue existing in ``data``, w.r.t. the specified user.
    :param model: The model used to compute the user-venue scores
    :type model: CheckinScorer
    :param data: The whole graph data
    :type data: HeteroData
    :param user_node_id: The node ID of the user we're computing the recommendations for
    :type user_node_id: int
    :param recap: Indicates wether to print the mean and standard deviation of the scores for debugging purposes.
    :type recap: bool
    :return: a dictionary holding the (venue_id, score) pairs
    :rtype: dict[int, float]
    """
    model.eval()

    with torch.no_grad():
        x_dict = {
            "user": model.user_emb(data["user"].node_id),
            "venue": model.venue_lin(data["venue"].x) + model.venue_emb(data["venue"].node_id)
        }

        x_dict = model.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)

        x_user = x_dict['user']
        x_venue = x_dict['venue']

        venue_node_ids = data["venue"].node_id.tolist()

        user_embedding = x_user[user_node_id].unsqueeze(0)
        venues_embeddings = x_venue[venue_node_ids]

        user_scores = LinkPredictor.decode(user_embedding, venues_embeddings)

    if recap:
        POILog.d(tag, f"Mean user score: {user_scores.mean().item()}")
        POILog.d(tag, f"User scores Standard Deviation: {user_scores.std().item()}")

    user_scores = [(venue_node_id, score.item()) for venue_node_id, score in zip(venue_node_ids, user_scores)]
    user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
    return dict(user_scores)


def evaluate_descriptions_scores(query_emb, descriptions_emb, device, recap=False):
    """
    Computes the description query-venue description similarity score for each element of ``description_emb`` against
    ``query_emb``.
    :param query_emb: The embedding of the description query
    :type query_emb: Tensor
    :param descriptions_emb: A dictionary containing the (venue_id, venue_embedding) pairs of all existing venues
    :type descriptions_emb: dict[str, Tensor]
    :param device: The device (CPU/GPU) where we should make the computations
    :type device: device
    :param recap: Indicates wether to print the mean and standard deviation of the scores for debugging purposes.
    :type recap: bool
    :return: a dictionary holding the (venue_id, score) pairs
    :rtype: dict[int, float]
    """
    venue_node_ids = descriptions_emb.keys()

    query_embedding = torch.tensor(query_emb, device=device).unsqueeze(0)
    desc_embeddings = torch.stack([torch.tensor(desc_emb, device=device) for desc_emb in descriptions_emb.values()])

    desc_scores = LinkPredictor.decode(query_embedding, desc_embeddings)

    if recap:
        POILog.d(tag, f"Mean description score: {desc_scores.mean().item()}")
        POILog.d(tag, f"Descriptions scores Standard Deviation: {desc_scores.std().item()}")

    desc_scores = [(venue_node_id, score.item()) for venue_node_id, score in zip(venue_node_ids, desc_scores)]
    desc_scores = sorted(desc_scores, key=lambda x: x[1], reverse=True)
    return dict(desc_scores)


def evaluate_total_scores(user_scores, desc_scores, user_weight=0.5, desc_weight=0.5, recap=False):
    """
    Computes a linear combination of the ``user_scores`` and ``desc_scores``, obtaining a total score which takes into
    account both the user compatibility and the query compatibility for each existing venue.
    :param user_scores: A dictionary containing the (venue_id, user_score) pairs of all existing venues
    :type user_scores: dict[int, float]
    :param desc_scores: A dictionary containing the (venue_id, desc_score) pairs of all existing venues
    :type desc_scores: dict[int, float]
    :param user_weight: The weight of the user scores component in the linear combination
    :type user_weight: float
    :param desc_weight: The weight of the user scores component in the linear combination
    :type desc_weight: float
    :param recap: Indicates wether to print the mean and standard deviation of the scores for debugging purposes.
                    NOTE: This feature is not yet implemented
    :type recap: bool
    :return: a dictionary holding the (venue_id, total_score) pairs
    :rtype: dict[int, float]
    """
    assert all([user_vid in desc_scores.keys() for user_vid in user_scores.keys()])
    venues_ids = user_scores.keys()

    total_scores = [
        (venue_id, user_weight * user_scores[venue_id] + desc_weight * desc_scores[venue_id])
        for venue_id in venues_ids
    ]

    if recap:
        POILog.d(tag, f"Not implemented")
        # POILog.d(tag, f"Mean total score: {total_scores.mean().item()}")
        # POILog.d(tag, f"Total scores Standard Deviation: {user_scores.std().item()}")

    sorted_total_scores = sorted(total_scores, key=lambda x: x[1], reverse=True)
    return dict(sorted_total_scores)
