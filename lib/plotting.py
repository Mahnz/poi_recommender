import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx
from config import PROJECT_ROOT

tag = "Plotting"


def visualize_graph_plotly(graph: nx.Graph, edge_scores=None):
    venue_categories = pd.read_csv(f"{PROJECT_ROOT}/Dataset/venue_categories.csv")
    pos = nx.spring_layout(graph, seed=42)

    edge_x = []
    edge_y = []
    edge_colors = []

    if edge_scores:
        # Normalizza i punteggi tra 0 e 1 per applicare la colormap
        scores_list = list(edge_scores.values())
        min_score = min(scores_list)
        max_score = max(scores_list)
        norm_edge_scores = {k: (v - min_score) / (max_score - min_score) for k, v in edge_scores.items()}
    else:
        norm_edge_scores = None

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        if norm_edge_scores:
            score = norm_edge_scores.get((edge[0], edge[1]), 0)
            color = plt.cm.viridis(score)
            edge_colors.append(f'rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, {color[3]})')
        else:
            edge_colors.append('#888')

    edge_traces = []
    for i in range(len(edge_x) // 3):
        edge_trace = go.Scatter(
            x=edge_x[i * 3:(i + 1) * 3],
            y=edge_y[i * 3:(i + 1) * 3],
            line=dict(width=1.5, color=edge_colors[i]),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []

    n_user_nodes = len([node for node, data_g in graph.nodes(data=True) if data_g.get('type') == 'user'])
    if n_user_nodes == 1:
        user_node_size = 100
        venue_node_size = 40
    else:
        user_node_size = 30
        venue_node_size = 10

    for node, data_g in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if data_g.get('type') == 'venue':
            node_size.append(venue_node_size + 0.5 * nx.degree(graph, node))
            node_color.append('#1f78b4')  # Blu per i venue
            category = int(data_g['x'][0])
            category = venue_categories[venue_categories['Cat_code'] == category]['Venue_category'].values[0]

            if edge_scores is not None:
                node_score = edge_scores.get((0, node), 0)
                node_text.append(
                    f'Node: {node}<br>Type: {data_g.get("type", "unknown")}<br>Category: {category}<br>Score: {node_score:.2f}')
            else:
                node_text.append(f'Node: {node}<br>Type: {data_g.get("type", "unknown")}<br>Category: {category}')
        elif data_g.get('type') == 'user':
            node_size.append(user_node_size + 0 * nx.degree(graph, node))
            node_color.append('#a71808')  # Rosso per gli user
            node_text.append(f'Node: {node}<br>Type: {data_g.get("type", "unknown")}')
        else:
            node_color.append('#b2df8a')
            node_text.append(f'Node: {node}<br>Type: {data_g.get("type", "unknown")}')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_color,
            size=node_size,
            line_width=2
        )
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))

    fig.show()


def create_nx_subgraph(subgraph):
    return to_networkx(
        subgraph,
        to_undirected=False,
        node_attrs=["x"],
    )


def plot_metric(metrics, metric_name,  colors, save_img=False):
    plt.figure(figsize=(10, 8))

    for (name, history), color in zip(metrics.items(), colors):
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, history, label=name, color=color)

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Epoch')
    plt.legend()
    plt.grid(True)

    if save_img:
        path = f"{PROJECT_ROOT}/plots"
        os.makedirs(path, exist_ok=True)
        path = f"{path}/{metric_name}.png"
        plt.savefig(path)
        print(f"Plot of {metric_name} saved as: '{path}'")

    plt.show()
