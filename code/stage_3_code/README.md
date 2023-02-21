# Overview

The code still follows the template, so I will personally document everything you should know, about where it deviates.

The code deviates from the template in four major ways:
1. Poetry package manager
2. Experimenting tracker with comet
    - think of this like tensorboard, but for a group
	- metric logging
	- experiment tracker acts as a subscriber/observer to various notification messages
		- handler is responsible for logging metrics
		- abstractions allow for extensible message types
3. setting configuration for hyperparameters
4. metric implementation:
	- classification metrics provided by the `torchmetrics` module:
		- accuracy, f1, recall, precision

# Getting Started

To get the code running please download the package manager [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer). Run `poetry shell` and verify your environment with `poetry env info`. Then, run `poetry install` to capture all the dependencies.


To run the code yourself, use the following command:
```
make stage_2
```

However, if you are not on a mac/linux, you can manually invoke `poetry run python -m script.stage_2_script.stage_2_script_mlp` and `poetry run python -m script.stage_2_script.stage_2_load_result`. Additionally, you will find a jupyter notebook in this file describing the data.
