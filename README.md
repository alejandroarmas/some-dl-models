We are using `poetry` as our package manager solution. To get started, run `poetry shell`, to create a nested shell with a virtual environment. Verify your environment with `poetry env info`. Then, run `poetry install` to capture all the dependencies. 



To use Comet, make sure to use the environment variable found [here]()
```
export COMET_API_KEY=<paste_your_key_here>
```

type checking library like MyPy and use annotations to ask for hard guarantees that all of the objects processed by your code — both container and contents — implement the runtime behaviors that your code requires

To use static type checking on a particular file, use for example:
```python
mypy code/lib/comet_listeners.py
```

To use linter on a particular file:
```python
pylint code/lib/evaluate_notifier.py
```