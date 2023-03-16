from code.base_class.dataset import datasetConfig
from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from typing import TypedDict

import networkx as nx  # type: ignore
import numpy as np
import torch
from matplotlib import pyplot as plt  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore


class visualizerConfig(TypedDict):
    output_folder_path: str
    output_image_name: str


class GraphVisualizer:
    def __init__(self, d_config: datasetConfig, v_config: visualizerConfig):
        self.data_obj = Dataset_Loader(d_config).load()
        self.output_folder = v_config["output_folder_path"]
        self.output_image = v_config["output_image_name"]
        if d_config["name"] == "cora":
            self.label_dict = {
                0: "Theory",
                1: "Reinforcement_Learning",
                2: "Genetic_Algorithms",
                3: "Neural_Networks",
                4: "Probabilistic_Methods",
                5: "Case_Based",
                6: "Rule_Learning",
            }
        elif d_config["name"] == "citeseer":
            self.label_dict = {0: "Agents", 1: "AI", 2: "DB", 3: "IR", 4: "ML", 5: "HCI"}
        elif d_config["name"] == "pubmed":
            self.label_dict = {0: "0", 1: "1", 2: "2"}
        else:
            raise Exception(
                f"{d_config['name']} is unknown. Known datasets: 'cora', 'citeseer', 'pubmed'"
            )

    def visualize(self):
        """Visualize given graph dataset and save the visualization in the given output image"""
        # Convert text loaded data to Data obj
        data = Data(
            x=torch.tensor(self.data_obj["graph"]["X"]),
            y=torch.tensor(self.data_obj["graph"]["y"]),
            edge_index=torch.tensor(np.transpose(self.data_obj["graph"]["edge"])),
        )
        print(
            f"Visualizing graph with {data.num_nodes} nodes, {data.num_edges} edges, and {data.num_features} features..."
        )
        # Convert to Network obj
        G = to_networkx(data, to_undirected=True)
        # Plot
        plt.figure(figsize=(12, 12))
        plt.axis("off")
        # Colors/Labels
        node_color = []
        nodelist = [[], [], [], [], [], [], []]
        color_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]
        labels = data.y
        # Node positions
        pos = nx.spring_layout(G, seed=42)
        for n, i in enumerate(labels):
            node_color.append(color_list[i])
            nodelist[i].append(n)
        label_list = list(self.label_dict.values())
        for num, i in enumerate(zip(nodelist, label_list)):
            n, label = i[0], i[1]
            nx.draw_networkx_nodes(
                G, pos, nodelist=n, node_size=5, node_color=color_list[num], label=label
            )
        nx.draw_networkx_edges(G, pos, width=0.25)
        plt.legend(loc="upper left", fontsize=10)
        plt.savefig(f"{self.output_folder}/{self.output_image}")
        print(f"Saved to {self.output_folder}/{self.output_image}")
