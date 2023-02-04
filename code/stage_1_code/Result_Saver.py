"""
Concrete ResultModule class for a specific experiment ResultModule output
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
from code.base_class.result import result
from code.lib.notifier import MLEventType, ResultNotification

import torch


class Result_Saver(result):
    fold_count = None

    def save(self):
        print("saving results...")
        filename = f"{self.result_destination_folder_path}{self.result_destination_file_name}_{str(self.fold_count)}"
        artifact_filename = f"{self.result_destination_folder_path}{self.result_destination_file_name}_{str(self.fold_count)}_artifacts"

        # save artifacts if parent model was given
        if self.parent_model is not None:
            torch.save(self.parent_model.state_dict(), artifact_filename)

        with open(filename, "wb") as f:
            pickle.dump(self.data, f)
            if self._manager is not None:
                # only passes parent_model.state_dict() if parent_model exists
                self._manager.notify(
                    MLEventType("save"),
                    ResultNotification(
                        self.data,
                        filename,
                        artifact_filename,
                        self.parent_model.state_dict() if self.parent_model is not None else None,
                    ),
                )

    def load(self) -> None:
        pass
