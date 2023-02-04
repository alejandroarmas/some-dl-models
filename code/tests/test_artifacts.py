import os
import unittest
from code.base_class.method import methodConfig
from code.base_class.result import resultConfig
from code.stage_1_code.Method_MLP import Method_MLP
from code.stage_1_code.Result_Saver import Result_Saver
from collections import OrderedDict
from unittest.mock import Mock

import torch
from torch import tensor

"""
    To run this class, perform `python -m code.tests.test_example`
    in the command line.
"""


class TestArtifacts(unittest.TestCase):
    def test_artifacts(self):
        r_config = resultConfig(
            {
                "name": "test",
                "description": "...data description...",
                "destination_folder_path": "code/tests/",
                "destination_file_name": "testartifacts",
            }
        )
        m_config = methodConfig(
            {
                "name": "test-method",
                "description": "This is a test",
            }
        )

        method_obj = Mock()
        testdict = OrderedDict(
            [
                (
                    "fc_layer_1.weight",
                    tensor(
                        [
                            [0.7172, 0.1651, -0.0167, 0.0633],
                            [0.2136, 0.1190, -0.0575, -0.4042],
                            [0.1142, -0.4427, 0.0657, 0.0332],
                            [-0.6775, 0.1094, 0.1887, 0.1240],
                        ]
                    ),
                ),
                ("fc_layer_1.bias", tensor([0.0213, -0.2950, -0.1922, 0.6772])),
                (
                    "fc_layer_2.weight",
                    tensor(
                        [[-1.1560, -0.0340, -0.0396, 0.9381], [0.6188, 0.1317, -0.0240, -0.8634]]
                    ),
                ),
                ("fc_layer_2.bias", tensor([-0.1579, -0.3684])),
            ]
        )

        method_obj.state_dict.return_value = testdict

        result_obj = Result_Saver(r_config, None, method_obj)
        result_obj.get_data([1, 2, 3, 4, 5])

        result_obj.save()

        new_model = Method_MLP(m_config, None)
        new_model.load_state_dict(
            torch.load(
                f"{result_obj.result_destination_folder_path}{result_obj.result_destination_file_name}_{str(result_obj.fold_count)}_artifacts"
            )
        )
        if os.path.exists(
            f"{result_obj.result_destination_folder_path}{result_obj.result_destination_file_name}_{str(result_obj.fold_count)}_artifacts"
        ):
            os.remove(
                f"{result_obj.result_destination_folder_path}{result_obj.result_destination_file_name}_{str(result_obj.fold_count)}_artifacts"
            )
        if os.path.exists(
            f"{result_obj.result_destination_folder_path}{result_obj.result_destination_file_name}_{str(result_obj.fold_count)}"
        ):
            os.remove(
                f"{result_obj.result_destination_folder_path}{result_obj.result_destination_file_name}_{str(result_obj.fold_count)}"
            )
        for a, b in zip(new_model.state_dict(), testdict):
            assert torch.equal(new_model.state_dict()[a], testdict[b])


if __name__ == "__main__":
    unittest.main()
