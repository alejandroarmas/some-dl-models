import os

# type: ignore
import cv2

# type: ignore
import matplotlib.pyplot as plt
import numpy as np
import onnx
import torch.nn as nn

# type: ignore
from onnx2pytorch import ConvertModel

# type: ignore
from torchvision import transforms


class CNNVisualizer:
    def __init__(self, onnx_file):
        self.filename = onnx_file
        onnx_model = onnx.load(self.filename)
        # Check model
        onnx.checker.check_model(onnx_model)
        # Convert to PyTorch
        self.model = ConvertModel(onnx_model)
        self.path = ("\\".join((os.path.abspath(os.getcwd()).split("\\")[:-2]))) + "\\"
        # Get all Conv layers & weights
        self.conv_layers = []
        model_children = list(self.model.children())
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                self.conv_layers.append(model_children[i])

    def visualize_filters(self):
        try:
            # Load first layer & weights
            conv_layer = self.conv_layers[0]
            weights = conv_layer.weight
            # Plot filters (max 8x8) & save in images->stage_3_images->filters
            rows = len(weights) // 9 + 1
            cols = len(weights) % 8 + 1
            image_path = (
                f"{self.path}images\\stage_3_images\\filters\\{self.filename.split('.')[0]}.png"
            )
            plt.figure(figsize=(20, 17))
            for i, filter in enumerate(weights):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(filter[0, :, :].detach(), cmap="gray")
                plt.axis("off")
            plt.savefig(image_path)
            print(f"Saved filters to {image_path}")
        except Exception as E:
            raise Exception(f"Visualize Filters Error: {E}")

    def visualize_activations(self, input_image, to_greyscale=True):
        try:
            # Read input image
            img = cv2.imread(f"{self.path}\\images\\stage_3_images\\input\\{input_image}")
            # Transforms
            transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            if to_greyscale:
                img = np.array(img)[:, :, 0]
            else:
                img = np.array(img)
            # Apply Transforms
            img = transform(img)
            # Add a batch dimension (3 x 28 x 28) -> (1 x 3 x 28 x 28)
            img = img.unsqueeze(0)
            # Load first layer & weights
            conv_layer = self.conv_layers[0]
            n_filters = len(conv_layer.weight)
            # Pass input image through layer
            result = conv_layer(img)
            layer_viz = result[0, :, :, :]
            layer_viz = layer_viz.data
            rows = n_filters // 9 + 1
            cols = n_filters % 8 + 1
            image_path = f"{self.path}images\\stage_3_images\\feature_maps\\{self.filename.split('.')[0]}_layer_0.png"
            plt.figure(figsize=(30, 30))
            for i, filter in enumerate(layer_viz):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(filter, cmap="gray")
                plt.axis("off")
            plt.savefig(image_path)
            print(f"Saved feature maps to {image_path}")
        except Exception as E:
            raise Exception(f"Visualize activations error: {E}")
