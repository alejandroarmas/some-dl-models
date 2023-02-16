import os
from code.base_class.artifacts import artifactConfig
from code.base_class.dataset import datasetConfig
from code.base_class.evaluate import EvaluateConfig
from code.base_class.method import methodConfig
from code.base_class.result import resultConfig
from code.base_class.setting import SettingConfig
from code.lib.comet_listeners import CometConfig, CometExperimentTracker
from code.lib.notifier import (
    DatasetNotifier,
    EvaluateNotifier,
    MethodNotifier,
    MLEventType,
    ResultNotifier,
    SettingNotifier,
)
from code.lib.notifier.artifacts_notifier import ArtifactsNotifier
from code.stage_1_code.Dataset_Loader import Dataset_Loader
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_1_code.Method_DT import Method_DT
from code.stage_1_code.Result_Saver import Result_Saver
from code.stage_1_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Artifacts_Saver import Artifacts_Saver
from code.stage_2_code.onnx_encoder import ONNX

import numpy as np
import torch

# ---- Decision Tree script ----
if 1:
    device = torch.device("cpu")
    # ---- parameter section -------------------------------
    np.random.seed(1)
    # ------------------------------------------------------

    algorithm_type = "DT"

    config = CometConfig(
        {
            "api_key": os.environ["COMET_API_KEY"],
            "project_name": "some-dl-models",
            "workspace": "ecs189g",
        }
    )

    experiment_tracker = CometExperimentTracker(config)

    d_config = datasetConfig(
        {
            "name": "toy",
            "description": "...data description...",
            "source_folder_path": "data/stage_1_data/",
            "source_file_name": "toy_data_file.txt",
            "device": device,
        }
    )

    r_config = resultConfig(
        {
            "name": "toy",
            "description": "...data description...",
            "destination_folder_path": f"result/stage_1_result/{algorithm_type}_",
            "destination_file_name": "prediction_result",
        }
    )
    m_config = methodConfig(
        {
            "name": f"{algorithm_type}-method",
            "description": "This is a decision tree",
            "hyperparameters": {},
        }
    )
    s_config = SettingConfig(
        {
            "name": "Setting_KFold_CV",
            "description": "This setting enables us to divide our data in sections",
            "device": device,
        }
    )

    e_config = EvaluateConfig(
        {"name": "recall", "description": "This is my recall object evaluator"}
    )

    d_notifier = DatasetNotifier()
    d_notifier.subscribe(experiment_tracker.dataset_listener, MLEventType("load"))
    data_obj = Dataset_Loader(d_config, d_notifier)

    m_notifier = MethodNotifier()
    m_notifier.subscribe(experiment_tracker.method_listener, MLEventType("method"))
    method_obj = Method_DT(m_config, m_notifier)

    r_notifier = ResultNotifier()
    r_notifier.subscribe(experiment_tracker.result_listener, MLEventType("save"))
    result_obj = Result_Saver(r_config, r_notifier)

    s_notifier = SettingNotifier()
    s_notifier.subscribe(experiment_tracker.setting_listener, MLEventType("setting"))
    setting_obj = Setting_KFold_CV(s_config, s_notifier)

    e_notifier = EvaluateNotifier()
    e_notifier.subscribe(experiment_tracker.evaluation_listener, MLEventType("evaluate"))
    evaluate_obj = Evaluate_Accuracy(e_config, e_notifier)

    a_config = artifactConfig(
        {
            "folder_path": "result/stage_2_artifacts/",
            "model_name": "sample_model",
            "input_dim": m_config["hyperparameters"]["input_dim"],
            "batch_size": 1,
            "output_dim": m_config["hyperparameters"]["output_dim"],
        }
    )

    a_notifier = ArtifactsNotifier()
    a_notifier.subscribe(experiment_tracker.artifacts_listener, MLEventType("save_artifacts"))
    # Uses the ONNX format for encoding our model artifacts
    artifact_encoder = ONNX(a_config, None)
    # Wraps the encoder object for comet integration
    artifact_obj = Artifacts_Saver(artifact_encoder, a_notifier)
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print("************ Start ************")
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj, artifact_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print("************ Overall Performance ************")
    print("Decision Tree Accuracy: " + str(mean_score) + " +/- " + str(std_score))
    print("************ Finish ************")
    # ------------------------------------------------------
