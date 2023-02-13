import abc
from code.base_class.artifacts import artifacts
from code.base_class.notifier import MLEventType
from code.lib.notifier.artifacts_notifier import ArtifactsNotification

from torch import nn

"""
Wrapper class for any artifacts object supporting encoding data. Calls to serialize() will run the wrapped objects
serialization function while also logging the saved file to Comet.
"""


class Artifacts_Saver(artifacts):
    encoder: artifacts

    def __init__(self, encoder: artifacts):
        assert isinstance(encoder, artifacts)
        self.__dict__ = encoder.__dict__  # Wraps all attempted access to encoder instance variables
        self.encoder = encoder

    # Calls the wrapped object's serialize method and then pings Comet with the filename
    def serialize(self):
        print("saving artifacts...")
        self.encoder.serialize()
        filename = f"{self.folder_path}{self.model_name}{self.extenson}"
        if self._manager is not None:
            self._manager.notify(MLEventType("save_artifacts"), ArtifactsNotification(filename))

    @abc.abstractmethod
    def deserialize(self) -> nn.Module:
        pass
