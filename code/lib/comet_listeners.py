from code.base_class.notifier import MLEventListener
from code.lib.notifier import (
    DatasetNotification,
    EvaluateNotification,
    MethodNotification,
    ResultNotification,
    SettingNotification,
)
from typing import Optional, TypedDict

from comet_ml import Experiment

# from comet_ml.integration.pytorch import log_model


class CometMethodHandler(MLEventListener):

    __experiment: Experiment
    __step: int

    def __init__(self, experiment: Optional[Experiment] = None):
        if experiment is not None:
            assert isinstance(experiment, Experiment)
        self.__experiment = experiment
        self.__step = 1

    def update(self, data: MethodNotification) -> None:

        data_report = {k: v for k, v in data.dict().items() if k not in {"epoch"}}

        if self.__experiment is not None:
            with self.__experiment.test():

                self.__experiment.log_metrics(data_report, step=self.__step, epoch=data.epoch)

                self.__step = self.__step + 1
        else:
            print(f"data_report: {data_report}")


class CometEvaluateHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Optional[Experiment] = None):
        if experiment is not None:
            assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: EvaluateNotification) -> None:

        if self.__experiment is not None:
            self.__experiment.log_parameter("measure", data.measure)
            self.__experiment.log_metric("evaluation", data.evaluation)
        else:
            print(f"measure: {data.measure}")
            print(f"evaluation: {data.evaluation}")


class CometResultHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Optional[Experiment] = None):
        if experiment is not None:
            assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: ResultNotification) -> None:
        if self.__experiment is not None:
            self.__experiment.log_parameter("filename", data.filename)


class CometSettingHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Optional[Experiment] = None):
        if experiment is not None:
            assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: SettingNotification) -> None:
        if self.__experiment is not None:
            self.__experiment.log_parameter("settings", data.config_settings)


class CometDatasetHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Optional[Experiment] = None):
        if experiment is not None:
            assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: DatasetNotification) -> None:
        if self.__experiment is not None:
            self.__experiment.log_parameters(
                {"filename": data.filename, "file_size": data.examples_size}
            )


class CometConfig(TypedDict):
    api_key: str
    project_name: str
    workspace: str


class CometExperimentTracker:
    method_listener: CometMethodHandler
    evaluation_listener: CometEvaluateHandler
    result_listener: CometResultHandler
    setting_listener: CometSettingHandler
    dataset_listener: CometDatasetHandler
    experiment: Experiment

    def __init__(self, config: CometConfig, dry_run: bool = False):

        if not dry_run:
            self.experiment = Experiment(
                api_key=config["api_key"],
                project_name=config["project_name"],
                workspace=config["workspace"],
            )
        else:
            self.experiment = None

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
