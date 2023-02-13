import unittest
from code.base_class.artifacts import artifactConfig
from code.base_class.method import methodConfig
from code.stage_2_code.onnx_encoder import ONNX

import onnx

"""
    To run this class, perform `python -m code.tests.test_example`
    in the command line.
"""


class TestArtifacts(unittest.TestCase):
    # Loads in a sample model from memory exported from an earlier experiment and checks its structural integrity
    def test_artifacts(self):
        m_config = methodConfig(
            {
                "name": "test-method",
                "description": "This is a multilayer perceptron",
                "hyperparameters": {
                    "max_epoch": 500,
                    "learning_rate": 5e-3,
                    "input_dim": 784,
                    "hidden_dim_0": 256,
                    "hidden_dim_1": 64,
                    "output_dim": 10,
                },
            }
        )
        a_config_0 = artifactConfig(
            {
                "folder_path": "code/tests/testfiles/",
                "model_name": "sample_model",
                "input_dim": m_config["hyperparameters"]["input_dim"],
                "batch_size": 1,
                "output_dim": m_config["hyperparameters"]["output_dim"],
            }
        )

        artifact_encoder_0 = ONNX(a_config_0, None, None)
        method_obj_0 = artifact_encoder_0.deserialize()

        onnx.checker.check_model(method_obj_0)


if __name__ == "__main__":
    unittest.main()
