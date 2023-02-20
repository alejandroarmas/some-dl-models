import os
from code.base_class.artifacts import artifactConfig
from code.base_class.dataset import datasetConfig
from code.base_class.evaluate import EvaluateConfig
from code.base_class.method import methodConfig
from code.base_class.result import resultConfig
from code.base_class.setting import SettingConfig
from code.lib.comet_listeners import CometConfig, CometExperimentTracker
from code.lib.encoding.Artifacts_Saver import Artifacts_Saver
from code.lib.encoding.onnx_encoder import ONNX
from code.lib.notifier import (
    ArtifactsNotifier,
    DatasetNotifier,
    EvaluateNotifier,
    MethodNotifier,
    MLEventType,
    ResultNotifier,
    SettingNotifier,
)

# from code.lib.util.device import get_device
from code.stage_3_code.Dataset_Loader import ValidatedPickleLoader
from code.stage_3_code.Evaluate_F1 import Evaluate_F1
from code.stage_3_code.Method_CNN_ORL import MethodCNN_ORL
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test_Split_ORL import (
    Setting_Train_Test_Split_ORL,
)
from dataclasses import dataclass

import numpy as np
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

# ---- Convolutional - Neural - Network script ----


@dataclass
class MasterConfig:
    c_config: CometConfig
    d_config: datasetConfig
    r_config: resultConfig
    m_config: methodConfig
    s_config: SettingConfig
    e_config: EvaluateConfig
    a_config: artifactConfig


def main():

    device: torch.device = torch.device("cpu")

    algorithm_type = "CNN"
    dataset_name = "ORL"

    config = CometConfig(
        {
            "api_key": os.environ["COMET_API_KEY"],
            "project_name": "some-dl-models",
            "workspace": "ecs189g",
        }
    )

    d_config = datasetConfig(
        {
            "name": dataset_name,
            "description": "dataset of something",
            "source_folder_path": "data/stage_3_data/",
            "source_file_name": dataset_name,
            "device": device,
        }
    )

    r_config = resultConfig(
        {
            "name": f"{dataset_name}-{algorithm_type}-result",
            "description": "...data description...",
            "destination_folder_path": f"result/stage_3_result/{algorithm_type}2_",
            "destination_file_name": "prediction_result",
        }
    )

    m_config = methodConfig(  # 84.5
        {
            "name": f"{algorithm_type}-method",
            "description": "This is a convolutional neural network",
            "hyperparameters": {
                "max_epoch": 72,  # was 36
                "learning_rate": 1e-3,
                "image_size": 112,
                "image_size2": 92,
                "conv_channels_in_dim": 1,
                "conv_channels_out_dim_0": 3,
                "conv_channels_out_dim_1": 5,
                "conv_kernel_size": 3,
                "pool_kernel_size": 2,
                "pool_stride": 2,
                "batch_size": 10,
                "output_dim_0": 90,
                "output_dim_1": 40,
            },
        }
    )

    s_config = SettingConfig(
        {
            "name": "Setting_Train_Test_Split_ORL",
            "description": "This setting enables us to divide our data in sections",
            "device": device,
        }
    )

    e_config = EvaluateConfig(
        {"name": "recall", "description": "This is my recall object evaluator"}
    )

    a_config = artifactConfig(
        {
            "folder_path": "result/stage_3_artifacts/",
            "model_name": "ORL_CNN2",
            "input_dim": (1, 112, 92),
            "batch_size": 10,
            "output_dim": 40,
        }
    )

    config1 = MasterConfig(config, d_config, r_config, m_config, s_config, e_config, a_config)

    runExperiment(config1)


def runExperiment(masterConfig):
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------
    # ---- objection initialization setction ---------------
    algorithm_type = "CNN"

    # device: torch.device = get_device()
    device: torch.device = torch.device("cpu")

    config = masterConfig.c_config
    d_config = masterConfig.d_config
    m_config = masterConfig.m_config
    r_config = masterConfig.r_config
    s_config = masterConfig.s_config
    e_config = masterConfig.e_config
    a_config = masterConfig.a_config

    experiment_tracker = CometExperimentTracker(config, dry_run=False)

    d_notifier = DatasetNotifier()
    d_notifier.subscribe(experiment_tracker.dataset_listener, MLEventType("load"))
    data_obj = ValidatedPickleLoader(d_config, d_notifier)

    m_notifier = MethodNotifier()
    m_notifier.subscribe(experiment_tracker.method_listener, MLEventType("method"))
    batch_metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
            MulticlassF1Score(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
            MulticlassPrecision(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
            MulticlassRecall(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
        ]
    )

    method_obj = MethodCNN_ORL(m_config, m_notifier, batch_metrics)

    r_notifier = ResultNotifier()
    r_notifier.subscribe(experiment_tracker.result_listener, MLEventType("save"))
    result_obj = Result_Saver(r_config, r_notifier)

    s_notifier = SettingNotifier()
    s_notifier.subscribe(experiment_tracker.setting_listener, MLEventType("setting"))
    setting_obj = Setting_Train_Test_Split_ORL(s_config, s_notifier)

    e_notifier = EvaluateNotifier()
    e_notifier.subscribe(experiment_tracker.evaluation_listener, MLEventType("evaluate"))
    final_evaluation = Evaluate_F1(e_config, e_notifier)

    a_notifier = ArtifactsNotifier()
    a_notifier.subscribe(experiment_tracker.artifacts_listener, MLEventType("save_artifacts"))
    # Uses the ONNX format for encoding our model artifacts
    artifact_encoder = ONNX(a_config, method_obj)
    # Wraps the encoder object for comet integration
    artifact_obj = Artifacts_Saver(artifact_encoder, a_notifier)

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print("************ Start ************")
    setting_obj.prepare(data_obj, method_obj, result_obj, final_evaluation, artifact_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print("************ Overall Performance ************")
    print(f"{algorithm_type} Accuracy: " + str(mean_score) + " +/- " + str(std_score))
    print("************ Finish ************")
    # ------------------------------------------------------


if __name__ == "__main__":
    main()
