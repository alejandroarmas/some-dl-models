from code.base_class.dataset import datasetConfig
from code.stage_5_code.GraphVisualizer import GraphVisualizer, visualizerConfig

import torch

d_config = datasetConfig(
    {
        "name": "cora",
        "description": "",
        "source_folder_path": "data/stage_5_data/cora",
        "source_file_name": "N/A",
        "device": torch.device("cpu"),
    }
)

v_config = visualizerConfig(
    {"output_folder_path": "images/stage_5_images", "output_image_name": "cora.png"}
)
G = GraphVisualizer(d_config, v_config)
G.visualize()

d_config = datasetConfig(
    {
        "name": "citeseer",
        "description": "",
        "source_folder_path": "data/stage_5_data/citeseer",
        "source_file_name": "N/A",
        "device": torch.device("cpu"),
    }
)

v_config = visualizerConfig(
    {"output_folder_path": "images/stage_5_images", "output_image_name": "citeseer.png"}
)
G = GraphVisualizer(d_config, v_config)
G.visualize()

d_config = datasetConfig(
    {
        "name": "pubmed",
        "description": "",
        "source_folder_path": "data/stage_5_data/pubmed",
        "source_file_name": "N/A",
        "device": torch.device("cpu"),
    }
)

v_config = visualizerConfig(
    {"output_folder_path": "images/stage_5_images", "output_image_name": "pubmed.png"}
)
G = GraphVisualizer(d_config, v_config)
G.visualize()
