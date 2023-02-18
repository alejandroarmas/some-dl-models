from code.base_class.artifacts import artifactConfig, artifacts
from typing import Optional

import onnx
import torch
from torch import nn

"""
Class for handling encoding and decoding of models to the ONNX file format
"""


class ONNX(artifacts):
    def __init__(self, config: artifactConfig, model: Optional[torch.nn.Module] = None):
        super().__init__(config, ".onnx", model)

    def serialize(self) -> None:

        filename = f"{self.folder_path}{self.model_name}{self.extension}"
        dummy_X = torch.randn(self.batch_size, *self.input_dim)        
        input_names = ["actual_input"]
        output_names = ["output"]
        if self.model is not None:
            self.model.train(
                mode=False
            )  # Method_MLP overrides this method, but it has an originally different purpose. Lets rename that function in Method_MLP to avoid errors

        torch.onnx.export(
            self.model,
            dummy_X,
            filename,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
        )

    def deserialize(self) -> nn.Module:

        filename = f"{self.folder_path}{self.model_name}{self.extension}"
        model = onnx.load(filename)
        print(onnx.helper.printable_graph(model.graph))
        return model
