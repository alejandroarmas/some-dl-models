from code.base_class.dataset import datasetConfig
from code.stage_5_code.Graph_Loader import Graph_Loader

import torch

d_config = datasetConfig(
    {
        "name": "Cora",
        "description": "Graph of academic papers",
        "source_folder_path": "data/stage_5_data/cora",
        "source_file_name": "N/A",
        "device": torch.device("cpu"),
    }
)

d = Graph_Loader(d_config)
