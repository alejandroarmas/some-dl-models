### Developers

- "Alejandro Armas <armas@ucdavis.edu>",
- "Lia Hepler-Mackey <lmheplermackey@ucdavis.edu>",
- "Sachin Loecher <smloecher@ucdavis.edu>"
- "Matthew Schulz <mkschulz@ucdavis.edu>"

### Getting Started

We are using `poetry` as our package manager solution. To get started, run `poetry shell`, to create a nested shell with a virtual environment. Verify your environment with `poetry env info`. Then, run `poetry install` to capture all the dependencies.


To use Comet, make sure to use the environment variable found [here](https://www.comet.com/account-settings/apiKeys)
```
export COMET_API_KEY=<paste_your_key_here>
```

You should now be able to run a particular script with the following command:
```
poetry run python -m script.stage_1_script.script_mlp
```

### Testing and Static Analysis

We are using `mypy`, `flake8`, `isort`, `pre-commit`, and `black`,

Mypy does **Type checking** to verify that our code follows its own type annotations. We use annotations to ask for hard guarantees that all of the objects processed by your code — both container and contents — implement the runtime behaviors that your code requires.

To use static type checking on a particular file, use for example:
```python
mypy code/lib/comet_listeners.py
```

To unit test your code use `poetry run python -m unittest`. If you are interested in testing a particular file use:
```
poetry run python -m unittest code.tests.test_example
```

We are using `flake8` as a **style linter**, to point out issues that don't cause bugs, but make the code less readable or are not in line with style guides such as Python's PEP 8 document. Additionally, it is an **error linter**, which points out syntax errors or other code that will result in unhandled exceptions and crashes. `isort` similarly takes care of sorting python import statements.

To use this on a particular file:
```python
poetry run flake8 code/lib/notifier/evaluate_notifier.py
```

We are using `black` as a **code formatter**, to change the style of our code, (mostly revolving around proper whitespace) without affecting the behavior of the program.
```python
poetry run black code/lib/notifier/evaluate_notifier.py
```

Thankfully this is all automated as a series of pre-commit hooks. If you would like to manually run these, execute with `poetry run pre-commit run --all-files`. However, expect this to run any time you perform a commit.
