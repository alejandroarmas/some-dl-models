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
        if self.input_type == "tensor":
            dummy_X = torch.randn(self.batch_size, *self.input_dim)
        else:
            pass
        # dummy_X = [self.LOREM_IPSUM] * self.batch_size
        input_names = ["actual_input"]
        output_names = ["output"]
        if self.model is not None:
            self.model.train(
                mode=False
            )  # Method_MLP overrides this method, but it has an originally different purpose. Lets rename that function in Method_MLP to avoid errors
        dynamic_axes = {"input": {0: "batch_size"}, "output": {1: "batch_size"}}
        torch.onnx.export(
            self.model,
            dummy_X,
            filename,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            dynamic_axes=dynamic_axes,
        )

    def deserialize(self) -> nn.Module:

        filename = f"{self.folder_path}{self.model_name}{self.extension}"
        model = onnx.load(filename)
        print(onnx.helper.printable_graph(model.graph))
        return model

    # CONST
    LOREM_IPSUM = """Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime mollitia,
    molestiae quas vel sint commodi repudiandae consequuntur voluptatum laborum
    numquam blanditiis harum quisquam eius sed odit fugiat iusto fuga praesentium
    optio, eaque rerum! Provident similique accusantium nemo autem. Veritatis
    obcaecati tenetur iure eius earum ut molestias architecto voluptate aliquam
    nihil, eveniet aliquid culpa officia aut! Impedit sit sunt quaerat, odit,
    tenetur error, harum nesciunt ipsum debitis quas aliquid. Reprehenderit,
    quia. Quo neque error repudiandae fuga? Ipsa laudantium molestias eos
    sapiente officiis modi at sunt excepturi expedita sint? Sed quibusdam
    recusandae alias error harum maxime adipisci amet laborum. Perspiciatis
    minima nesciunt dolorem! Officiis iure rerum voluptates a cumque velit
    quibusdam sed amet tempora. Sit laborum ab, eius fugit doloribus tenetur
    fugiat, temporibus enim commodi iusto libero magni deleniti quod quam
    consequuntur! Commodi minima excepturi repudiandae velit hic maxime
    doloremque. Quaerat provident commodi consectetur veniam similique ad
    earum omnis ipsum saepe, voluptas, hic voluptates pariatur est explicabo
    fugiat, dolorum eligendi quam cupiditate excepturi mollitia maiores labore
    suscipit quas? Nulla, placeat. Voluptatem quaerat non architecto ab laudantium
    modi minima sunt esse temporibus sint culpa, recusandae aliquam numquam
    totam ratione voluptas quod exercitationem fuga. Possimus quis earum veniam
    quasi aliquam eligendi, placeat qui corporis!"""
