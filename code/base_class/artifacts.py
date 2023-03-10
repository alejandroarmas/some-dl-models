from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple, TypedDict

from torch import nn


class artifactConfig(TypedDict):
    folder_path: str
    model_name: str
    batch_size: int
    input_dim: Tuple[int, ...]
    output_dim: int
    input_type: Literal["tensor", "string"]


class artifacts(ABC):
    folder_path: str
    model_name: str
    extension: str
    batch_size: int
    input_dim: Tuple
    output_dim: int
    input_type: Literal["tensor", "string"]

    model: Optional[nn.Module]

    def __init__(self, config: artifactConfig, extension: str, model: Optional[nn.Module]):
        self.folder_path = config["folder_path"]
        self.model_name = config["model_name"]
        self.extension = (
            extension  # This should be passed from a subclass implementation, not by the end user
        )
        self.input_dim = config["input_dim"]
        self.batch_size = config["batch_size"]
        self.output_dim = config["output_dim"]
        self.model = model
        if "input_type" in config.keys():
            self.input_type = config["input_type"]
        else:
            self.input_type = "tensor"

    @abstractmethod
    def serialize(self):
        ...

    @abstractmethod
    def deserialize(self) -> nn.Module:
        ...
