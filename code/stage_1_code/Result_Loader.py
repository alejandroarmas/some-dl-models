"""
Concrete ResultModule class for a specific experiment ResultModule output
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
from code.base_class.result import result
from code.lib.notifier import MLEventType, ResultNotification


class Result_Loader(result):
    data = None

    def load(self) -> None:
        print("loading results...")
        filename = f"{self.result_destination_folder_path}{self.result_destination_file_name}_{str(self.fold_count)}"

        with open(filename, "rb") as f:
            self.data = pickle.load(f)
            if self._manager is not None:
                self._manager.notify(MLEventType("load"), ResultNotification(self.data, filename))

    def save(self) -> None:
        pass
