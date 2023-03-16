import os
from code.base_class.artifacts import artifactConfig
from code.base_class.dataset import datasetConfig
from code.base_class.evaluate import EvaluateConfig
from code.base_class.method import methodConfig
from code.base_class.notifier import MLEventType
from code.base_class.result import resultConfig
from code.base_class.setting import SettingConfig
from code.lib.comet_listeners import CometConfig, CometExperimentTracker
from code.lib.encoding.Artifacts_Saver import Artifacts_Saver
from code.lib.encoding.pytorch_encoder import torch_encoder
from code.lib.notifier import (
    ArtifactsNotifier,
    DatasetNotifier,
    EvaluateNotifier,
    MethodNotifier,
    ResultNotifier,
    SettingNotifier,
)
from code.stage_5_code.Citeseer_GCN import Citeseer_GCN
from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Evaluate_F1 import Evaluate_F1
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting_Equal_Sampling_Split import (
    Setting_Equal_Sampling_Split,
)

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

device: torch.device = torch.device("cpu")

algorithm_type = "GCN_Classification"
dataset_name = "citeseer"

if 1:
    config = CometConfig(
        {
            "api_key": os.environ["COMET_API_KEY"],
            "project_name": "some-dl-models",
            "workspace": "ecs189g",
        }
    )

    experiment_tracker = CometExperimentTracker(config, dry_run=False)

    d_config = datasetConfig(
        {
            "name": dataset_name,
            "description": "citation graph dataset",
            "source_folder_path": f"data/stage_5_data/{dataset_name}",
            "source_file_name": "N/A",  # not needed for this loader
            "device": device,
        }
    )

    s_config = SettingConfig(
        {
            "name": "Setting_Equal_Sampling_Split",
            "description": "Splits dataset into train/test/val sets with uniform class distribution in train and test",
            "device": device,
            "params": {
                "train_size": 120,
                "test_size": 1200,
            },
        }
    )
    r_config = resultConfig(
        {
            "name": f"{dataset_name}-{algorithm_type}-result",
            "description": "citation dataset of papers",
            "destination_folder_path": f"result/stage_5_result/{algorithm_type}_",
            "destination_file_name": "prediction_result",
        }
    )

    e_config = EvaluateConfig(
        {"name": "recall", "description": "This is my recall object evaluator"}
    )

    d_notifier = DatasetNotifier()
    d_notifier.subscribe(experiment_tracker.dataset_listener, MLEventType("load"))
    data_obj = Dataset_Loader(d_config, d_notifier)

    m_config = methodConfig(
        {
            "name": f"{algorithm_type}-method",
            "description": "This is a convolutional GNN",
            "hyperparameters": {
                "input_dim": 3703,
                "max_epoch": 31,
                "hidden_dim_1": 800,
                "hidden_dim_2": 200,
                "hidden_dim_3": 50,
                "output_dim": 6,
                "learning_rate": 1e-3,
            },
        }
    )

    m_notifier = MethodNotifier()
    m_notifier.subscribe(experiment_tracker.method_listener, MLEventType("method"))
    batch_metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=m_config["hyperparameters"]["output_dim"]).to(device),
            MulticlassF1Score(num_classes=m_config["hyperparameters"]["output_dim"]).to(device),
            MulticlassPrecision(num_classes=m_config["hyperparameters"]["output_dim"]).to(device),
            MulticlassRecall(num_classes=m_config["hyperparameters"]["output_dim"]).to(device),
        ]
    )
    method_obj = Citeseer_GCN(m_config, m_notifier, batch_metrics)

    r_notifier = ResultNotifier()
    r_notifier.subscribe(experiment_tracker.result_listener, MLEventType("save"))
    result_obj = Result_Saver(r_config, r_notifier)

    s_notifier = SettingNotifier()
    s_notifier.subscribe(experiment_tracker.setting_listener, MLEventType("setting"))
    setting_obj = Setting_Equal_Sampling_Split(s_config, s_notifier)

    e_notifier = EvaluateNotifier()
    e_notifier.subscribe(experiment_tracker.evaluation_listener, MLEventType("evaluate"))
    final_evaluation = Evaluate_F1(e_config, e_notifier)

    a_config = artifactConfig(
        {
            "folder_path": "result/stage_5_artifacts/",
            "model_name": "GCN_Citeseer_Classification",
            "batch_size": 1,
            "input_dim": (1, 50),
            "output_dim": m_config["hyperparameters"]["output_dim"],
            "input_type": "graph",
        }
    )

    a_notifier = ArtifactsNotifier()
    a_notifier.subscribe(experiment_tracker.artifacts_listener, MLEventType("save_artifacts"))
    # Uses the ONNX format for encoding our model artifacts
    artifact_encoder = torch_encoder(a_config, method_obj)
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
