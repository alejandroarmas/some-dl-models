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
from code.stage_4_code.Dataset_Loader_Classification import (
    Classification_Loader,
    Classification_Vocabulary,
)
from code.stage_4_code.Evaluate_F1 import Evaluate_F1
from code.stage_4_code.Method_GRU_classification import MethodGRUClassification

# from code.stage_4_code.Method_RNN_classification import MethodRNNClassification
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test_Split_RNN_Classififcation import (
    Setting_Train_Test_Split,
)
from typing import TypedDict

import numpy as np
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class metrics(TypedDict):
    train: MetricCollection
    test: MetricCollection


def main():
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------
    # ---- objection initialization setction ---------------
    device: torch.device = torch.device("cpu")

    algorithm_type = "RNN_Classification"
    dataset_name = "MovieReviews"

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
            "description": "dataset of film reviews, both positive and negative in sentiment",
            "source_folder_path": "data/stage_4_data/text_classification",
            "source_file_name": "N/A",
            "device": device,
        }
    )

    r_config = resultConfig(
        {
            "name": f"{dataset_name}-{algorithm_type}-result",
            "description": "dataset of movie reviews with both positive and negative sentiment",
            "destination_folder_path": f"result/stage_4_result/{algorithm_type}_",
            "destination_file_name": "prediction_result",
        }
    )

    s_config = SettingConfig(
        {
            "name": "Setting_Train_Test_Split",
            "description": "This setting enables us to divide our data in sections",
            "device": device,
        }
    )

    e_config = EvaluateConfig(
        {"name": "recall", "description": "This is my recall object evaluator"}
    )

    experiment_tracker = CometExperimentTracker(config, dry_run=True)

    d_notifier = DatasetNotifier()
    d_notifier.subscribe(experiment_tracker.dataset_listener, MLEventType("load"))
    data_obj = Classification_Loader(d_config, d_notifier)

    print("-------Building Vocabulary-------")
    cutoff_value = 100
    vocab_obj = Classification_Vocabulary(data_obj, cutoff_value)

    m_config = methodConfig(
        {
            "name": f"{algorithm_type}-method",
            "description": "This is a GRU Recursive Neural Network",
            "hyperparameters": {
                "input_size": 50,
                "hidden_size": 16,
                "num_layers": 1,
                "nonlinearity": "tanh",
                "dropout": 0,
                "dense_size_1": 8,
                "output_dim_1": 1,
                "learning_rate": 1e-3,
                "max_epoch": 4,
                "batch_size": 100,
                "vocab_size": vocab_obj.get_vocab().__len__(),
            },
        }
    )

    m_notifier = MethodNotifier()
    m_notifier.subscribe(experiment_tracker.method_listener, MLEventType("method"))
    train_batch_metrics = MetricCollection(
        [
            BinaryAccuracy(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
            BinaryF1Score(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
            BinaryPrecision(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
            BinaryRecall(num_classes=m_config["hyperparameters"]["output_dim_1"]).to(device),
        ]
    )
    method_obj = MethodGRUClassification(m_config, m_notifier, train_batch_metrics)

    r_notifier = ResultNotifier()
    r_notifier.subscribe(experiment_tracker.result_listener, MLEventType("save"))
    result_obj = Result_Saver(r_config, r_notifier)

    s_notifier = SettingNotifier()
    s_notifier.subscribe(experiment_tracker.setting_listener, MLEventType("setting"))
    setting_obj = Setting_Train_Test_Split(s_config, s_notifier)

    e_notifier = EvaluateNotifier()
    e_notifier.subscribe(experiment_tracker.evaluation_listener, MLEventType("evaluate"))
    final_evaluation = Evaluate_F1(e_config, e_notifier)

    a_config = artifactConfig(
        {
            "folder_path": "result/stage_4_artifacts/",
            "model_name": "RNN_Classification",
            "input_dim": (32, 768),
            "batch_size": 1,
            "output_dim": m_config["hyperparameters"]["output_dim_1"],
            "input_type": "string",
        }
    )

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
    setting_obj.prepare_vocab(vocab_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print("************ Overall Performance ************")
    print(f"{algorithm_type} Accuracy: " + str(mean_score) + " +/- " + str(std_score))
    print("************ Finish ************")
    # ------------------------------------------------------


if __name__ == "__main__":
    main()
