"""
A wrapper class for multiple different file encoding formats that will all be
run at once. Only supports saving classes, and will throw an exception if attempting to load
"""
from code.base_class.artifacts import artifacts
from typing import Tuple

from torch import nn


class multi_encoder(artifacts):
    encoders: Tuple[artifacts, ...]

    def __init__(self, *encoders):
        self.encoders = encoders

    def serialize(self):
        for enc in self.encoders:
            enc.serialize()

    def deserialize(self) -> nn.Module:
        raise Exception(
            "Cannot deserialize multiple encoding formats at once. Please deserialize individually"
        )
