import os
import unittest
from code.base_class.artifacts import artifactConfig
from code.base_class.method import methodConfig
from code.lib.encoding.onnx_encoder import ONNX
from code.stage_2_code.Method_MLP import Method_MLP

import numpy as np
import onnx
import pandas as pd

"""
Tests ONNX encoding and decoding of a small MLP
To perform this unit test run: "poetry run python -m unittest code.tests.test_artifacts"
"""


class TestArtifacts(unittest.TestCase):
    def test_artifacts(self):
        m_config = methodConfig(
            {
                "name": "test-method",
                "description": "This is a multilayer perceptron",
                "hyperparameters": {
                    "max_epoch": 10,
                    "learning_rate": 5e-3,
                    "input_dim": (10),
                    "hidden_dim_0": 8,
                    "hidden_dim_1": 4,
                    "output_dim": 2,
                },
            }
        )
        a_config_0 = artifactConfig(
            {
                "folder_path": "code/tests/testfiles/",
                "model_name": "smol_model",
                "input_dim": m_config["hyperparameters"]["input_dim"],
                "batch_size": 1,
                "output_dim": m_config["hyperparameters"]["output_dim"],
            }
        )
        method_obj_0 = Method_MLP(m_config, None, None)
        # Creates a small dummy dataset
        X = pd.DataFrame(np.random.randint(0, 255, size=(20, 10)))
        y = pd.Series(np.random.rand(20))

        method_obj_0.train_model(X, y)
        artifact_encoder_0 = ONNX(a_config_0, method_obj_0)
        artifact_encoder_0.serialize()

        method_obj_1 = artifact_encoder_0.deserialize()
        # checks to make sure the model is structurally sound. If not an exception is thrown
        onnx.checker.check_model(method_obj_1)
        if os.path.exists(
            f'{a_config_0["folder_path"]}{a_config_0["model_name"]}{artifact_encoder_0.extension}'
        ):
            os.remove(
                f'{a_config_0["folder_path"]}{a_config_0["model_name"]}{artifact_encoder_0.extension}'
            )


if __name__ == "__main__":
    unittest.main()
