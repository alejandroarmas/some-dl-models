'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.result import result
from code.lib.notifier import ResultNotification, MLEventType

import pickle


class Result_Loader(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def load(self):
        print('loading results...')
        filename = f'{self.result_destination_folder_path}{self.result_destination_file_name}_{str(self.fold_count)}' 

        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
            self._manager.notify(MLEventType('load'), ResultNotification(self.data, filename))
