from code.base_class.notifier import MLEventListener
from code.lib.notifier import (
    DatasetNotification,
    EvaluateNotification,
    MethodNotification,
    ResultNotification,
    SettingNotification,
)
from typing import TypedDict

from comet_ml import Experiment

# from comet_ml.integration.pytorch import log_model


class CometMethodHandler(MLEventListener):

    __experiment: Experiment
    __step: int

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment
        self.__step = 1

    def update(self, data: MethodNotification) -> None:

        with self.__experiment.test():

            data_report = {k: v for k, v in data.dict().items() if k not in {"epoch"}}

            self.__experiment.log_metrics(data_report, step=self.__step, epoch=data.epoch)

            self.__step = self.__step + 1


class CometEvaluateHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: EvaluateNotification) -> None:

        self.__experiment.log_parameter("measure", data.measure)
        self.__experiment.log_metric("evaluation", data.evaluation)


class CometResultHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: ResultNotification) -> None:
        self.__experiment.log_parameter("filename", data.filename)


class CometSettingHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: SettingNotification) -> None:
        self.__experiment.log_parameter("settings", data.config_settings)


class CometDatasetHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: DatasetNotification) -> None:
        self.__experiment.log_parameters(
            {"filename": data.filename, "file_size": data.examples_size}
        )


class CometConfig(TypedDict):
    api_key: str
    project_name: str
    workspace: str


class CometExperimentTracker:
    method_listener: CometMethodHandler
    evaluate_listener: CometEvaluateHandler
    result_listener: CometResultHandler
    setting_listener: CometSettingHandler
    dataset_listener: CometDatasetHandler
    experiment: Experiment = None

    def __init__(self, config: CometConfig):

        self.experiment = Experiment(
            api_key=config["api_key"],
            project_name=config["project_name"],
            workspace=config["workspace"],
        )

        self.method_listener = CometMethodHandler(self.experiment)
        self.evaluation_listener = CometEvaluateHandler(self.experiment)
        self.result_listener = CometResultHandler(self.experiment)
        self.setting_listener = CometSettingHandler(self.experiment)
        self.dataset_listener = CometDatasetHandler(self.experiment)

    def log_model(self) -> None:
        ...

    def log_evaluation(self) -> None:
        ...

    def log_dataset(self) -> None:
        ...

    def log_setting(self) -> None:
        ...

    def log_results(self) -> None:
        ...
