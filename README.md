### Developers

- "Alejandro Armas <armas@ucdavis.edu>", 
- "Lia Hepler-Mackey <lmheplermackey@ucdavis.edu>", 
- "Sachin Loecher <smloecher@ucdavis.edu>"

### Getting Started

We are using `poetry` as our package manager solution. To get started, run `poetry shell`, to create a nested shell with a virtual environment. Verify your environment with `poetry env info`. Then, run `poetry install` to capture all the dependencies. 


To use Comet, make sure to use the environment variable found [here]()
```
export COMET_API_KEY=<paste_your_key_here>
```

You should now be able to run a particular script with the following command:
```
poetry run python -m script.stage_1_script.script_mlp
```

### Testing and Static Analysis

type checking library like MyPy and use annotations to ask for hard guarantees that all of the objects processed by your code — both container and contents — implement the runtime behaviors that your code requires

To unit test your code use `poetry run python -m unittest`. If you are interested in testing a particular file use:
```
poetry run python -m unittest code.tests.test_example
```

Pylint enforces a coding standard and is also a general linter.
To use linter on a particular file:
```python
pylint code/lib/evaluate_notifier.py
```


MyPy is a static type checker for Python.

To use static type checking on a particular file, use for example:
```python
mypy code/lib/comet_listeners.py
```