from code.base_class.artifacts import artifactConfig, artifacts
from typing import Optional

import torch
from torch import nn

"""
Class for handling encoding and decoding of models to the built-in pytorch file format
"""


class torch_encoder(artifacts):
    def __init__(self, config: artifactConfig, model: Optional[torch.nn.Module] = None):
        super().__init__(config, ".pth", model)

    def serialize(self) -> None:
        filename = f"{self.folder_path}{self.model_name}{self.extension}"
        if self.model is not None:
            torch.save(self.model, filename)

    def deserialize(self) -> nn.Module:
        filename = f"{self.folder_path}{self.model_name}{self.extension}"
        model = torch.load(filename)
        return model
