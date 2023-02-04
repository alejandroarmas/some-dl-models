from code.base_class.result import resultConfig
from code.stage_1_code.Result_Loader import Result_Loader

if 1:

    experiment_type = "MLP"

    r_config = resultConfig(
        {
            "name": "loader",
            "description": "this is used to load our experiment results locally",
            "destination_folder_path": f"result/stage_1_result/{experiment_type}_",
            "destination_file_name": "prediction_result",
        }
    )

    result_obj = Result_Loader(r_config)

    result_obj.fold_count = None
    result_obj.load()
    print("Result:", result_obj.data)
