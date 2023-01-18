from code.base_class.notifier import MLEventListener

from code.lib.notifier import (MethodNotification,
         DatasetNotification, ResultNotification, SettingNotification, EvaluateNotification)

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

from typing import TypedDict


class CometMethodHandler(MLEventListener):

    __experiment: Experiment
    __step: int

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment
        self.__step = 1

    def update(self, data: MethodNotification) -> None:

        with self.__experiment.test():

            self.__experiment.log_metrics({'accuracy': data.accuracy, 'loss': data.loss}, step=self.__step, epoch=data.epoch)
            self.__step = self.__step + 1

class CometEvaluateHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: EvaluateNotification) -> None:

        self.__experiment.log_parameter('measure', data.measure)
        self.__experiment.log_metric('evaluation', data.evaluation)
        print(f'data: EvaluateNotification')

class CometResultHandler(MLEventListener):
    
    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: ResultNotification) -> None:
        self.__experiment.log_parameter('filename', data.filename)

        print(f'data: ResultNotification')

class CometSettingHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: SettingNotification) -> None:
        self.__experiment.log_parameter('settings', data.config_settings)
        print(f'data: SettingNotification')


class CometDatasetHandler(MLEventListener):

    __experiment: Experiment

    def __init__(self, experiment: Experiment):
        assert isinstance(experiment, Experiment)
        self.__experiment = experiment

    def update(self, data: DatasetNotification) -> None:
        self.__experiment.log_parameters({'filename': data.filename, 'file_size': data.examples_size})
        print(f'data: DatasetNotification')



class CometConfig(TypedDict):
    api_key: str
    project_name: str
    workspace: str


class CometExperimentTracker():
    method_listener: CometMethodHandler
    evaluate_listener: CometEvaluateHandler
    result_listener: CometResultHandler
    setting_listener: CometSettingHandler
    dataset_listener: CometDatasetHandler
    experiment: Experiment = None

    def __init__(self, config: CometConfig):        
        
        self.experiment = Experiment(
            api_key=config['api_key'],
            project_name=config['project_name'],
            workspace=config['workspace'],
        )

        self.method_listener = CometMethodHandler(self.experiment)
        self.evaluation_listener = CometEvaluateHandler(self.experiment)
        self.result_listener = CometResultHandler(self.experiment)
        self.setting_listener = CometSettingHandler(self.experiment)
        self.dataset_listener = CometDatasetHandler(self.experiment)


    def log_model(self) -> None:
        print(f'Model has been logged')

    def log_evaluation(self) -> None:
        print(f'Evaluation has been logged')

    def log_dataset(self) -> None:
        print(f'dataset has been logged')

    def log_setting(self) -> None:
        print(f'setting has been logged')

    def log_results(self) -> None:
        print(f'results has been logged')