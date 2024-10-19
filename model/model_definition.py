import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate):
        super().__init__()

        self.conv1 = GATConv(hidden_channels, hidden_channels, edge_dim=1, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=1, add_self_loops=False)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class LinkPredictor(torch.nn.Module):
    def forward(self, x_user: Tensor, x_venue: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_venue = x_venue[edge_label_index[1]]

        return F.cosine_similarity(edge_feat_user, edge_feat_venue, dim=-1)

    @staticmethod
    def decode(emb1, emb2):
        similarities = F.cosine_similarity(emb1, emb2, dim=-1)
        scores = similarities.sigmoid()
        return scores


class CheckinScorer(torch.nn.Module):
    def __init__(self, num_users, num_venues, hidden_channels, metadata, dropout_rate):
        super().__init__()

        self.venue_lin = torch.nn.Linear(385, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.venue_emb = torch.nn.Embedding(num_venues, hidden_channels)

        self.gnn = GNN(hidden_channels, dropout_rate)  # Encoder
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')

        self.link_predictor = LinkPredictor()  # Decoder

    def forward(self, data: HeteroData) -> Tensor:
        # Embed
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "venue": self.venue_lin(data["venue"].x) + self.venue_emb(data["venue"].node_id)
        }

        # Encode
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)

        # Decode
        pred = self.link_predictor(
            x_dict["user"],
            x_dict["venue"],
            data["user", "checkin", "venue"].edge_label_index
        )

        return pred
