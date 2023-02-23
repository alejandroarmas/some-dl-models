# Overview

- The code deviates from the template in four major ways:
	1. Poetry package manager
	2. Experimenting tracker with Comet (think of this like tensorboard, but for a group), provides
		- metric logging
		- experiment tracker acts as a subscriber/observer to various notification messages
			- handler is responsible for logging metrics
			- abstractions allow for extensible message types
	3. setting configuration for hyperparameters
	4. metric implementation:
		- classification metrics provided by the `torchmetrics` module:
			- accuracy, f1, recall, precision

# Getting Started

- To get the code running please download the package manager [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer). Run `poetry shell` and verify your environment with `poetry env info`. Then, run `poetry install` to capture all the dependencies.

- Then, to run each CNN model in Stage 3, you can run one of the following commands:
	- to run the CNN model for the CIFAR dataset:
	```
	make stage_3_cifar
	```
	- to run the CNN model for the MNIST dataset:
	```
	make stage_3_mnist
	```
	- to run the CNN model for the ORL dataset
	```
	make stage_3_orl
	```

- If you are on Windows, run the following commands:
	- to run the CNN model for the CIFAR dataset:
	```
	poetry run python -m script.stage_3_script.cifar_cnn_script 
	```
	- to run the CNN model for the MNIST dataset:
	```
	poetry run python -m script.stage_3_script.mnist_cnn_script
	```
	- to run the CNN model for the ORL dataset
	```
	poetry run python -m script.stage_3_script.orl_cnn_script 
	```

# General Notes
## Stage 3 Scripts
- Each script in the `stage_3_script` folder utilizes a Comet experiment tracker, which will automatically record the models’ metrics to Comet when ran. Please note that each Comet experiment tracker can be passed a boolean argument named `dry_run` which will disable it from recording information. By default, each tracker’s `dry_run` argument is set to True, denoting that Comet will not log any information when the models are run


