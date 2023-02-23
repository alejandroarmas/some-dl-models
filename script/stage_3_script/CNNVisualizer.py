from abc import ABC, abstractmethod
from typing import TypedDict

import cv2  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import onnx
import torch.nn as nn
from onnx2pytorch import ConvertModel  # type: ignore
from torchvision import transforms  # type: ignore


class visualizerConfig(TypedDict):
    onnx_folder_path: str
    filter_image_output_path: str
    activations_image_output_path: str
    input_images_path: str


class Visualizer(ABC):
    def __init__(self, config: visualizerConfig):
        self.onnx_folder_path = config["onnx_folder_path"]
        self.filter_image_output_path = config["filter_image_output_path"]
        self.activations_image_output_path = config["activations_image_output_path"]
        self.input_images_path = config["input_images_path"]

    @abstractmethod
    def visualize(
        self,
        onnx_file: str,
        output_image_name: str,
    ):
        ...


class CNNFilterVisualizer(Visualizer):
    def visualize(self, onnx_filename: str, output_image_name: str) -> None:
        """Visualize learned filters of first convolutional layer of a given onnx model, and save the visualization
        in a given output image"""
        onnx_path = self.onnx_folder_path + onnx_filename
        onnx_model = onnx.load(onnx_path)
        # Check model
        onnx.checker.check_model(onnx_model)
        # Convert to PyTorch
        model = ConvertModel(onnx_model)
        # Get all Conv layers & weights
        conv_layers: list[nn.Conv2d] = []
        model_children = list(model.children())
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                conv_layers.append(model_children[i])
        # Load first layer & weights
        conv_layer = conv_layers[0]
        weights = conv_layer.weight
        # Plot filters (max 8x8) & save in filter_image_output_path
        rows = len(weights) // 9 + 1
        cols = len(weights) % 8 + 1
        plt.figure(figsize=(20, 17))
        for i, filter in enumerate(weights):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(filter[0, :, :].detach(), cmap="gray")
            plt.axis("off")
        image_path = self.filter_image_output_path + output_image_name
        plt.savefig(image_path)


class CNNActivationVisualizer(Visualizer):
    def __init__(
        self, config: visualizerConfig, input_image_name: str, input_to_greyscale: bool = True
    ):
        super().__init__(config)
        self.input_image_name = input_image_name
        self.input_to_greyscale = input_to_greyscale

    def visualize(
        self,
        onnx_filename: str,
        output_image_name: str,
    ) -> None:
        """Visualize activations/feature maps of the learned filters in the first convolutional layer of a given onnx model
        for a given input image, with the option to pass the input image as greyscale for models which only accept one
        channel."""
        onnx_path = self.onnx_folder_path + onnx_filename
        onnx_model = onnx.load(onnx_path)
        # Check model
        onnx.checker.check_model(onnx_model)
        # Convert to PyTorch
        model = ConvertModel(onnx_model)
        # Get all Conv layers & weights
        conv_layers: list[nn.Conv2d] = []
        model_children = list(model.children())
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                conv_layers.append(model_children[i])
        # Read input image
        img = cv2.imread(self.input_images_path + self.input_image_name)
        if self.input_to_greyscale:
            img = np.array(img)[:, :, 0]
        else:
            img = np.array(img)
        # To Tensor transform (so we can unsqueeze)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        # Add a batch dimension (3 x 28 x 28) -> (1 x 3 x 28 x 28)
        img = img.unsqueeze(0)
        # Load first layer & weights
        conv_layer = conv_layers[0]
        n_filters = len(conv_layer.weight)
        # Pass input image through layer
        result = conv_layer(img)
        layer_viz = result[0, :, :, :]
        layer_viz = layer_viz.data
        rows = n_filters // 9 + 1
        cols = n_filters % 8 + 1
        plt.figure(figsize=(30, 30))
        for i, filter in enumerate(layer_viz):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(filter, cmap="gray")
            plt.axis("off")
        image_path = self.activations_image_output_path + output_image_name
        plt.savefig(image_path)
