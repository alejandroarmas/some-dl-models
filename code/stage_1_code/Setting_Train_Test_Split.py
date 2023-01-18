'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self._dataset.load()

        X_train, X_test, y_train, y_test = train_test_split(loaded_data['X'], loaded_data['y'], test_size = 0.33)

        # run MethodModule
        self._method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self._method.run()
            
        # save raw ResultModule
        self._result.data = learned_result
        self._result.save()
            
        self._evaluate.data = learned_result
        
        return self._evaluate.evaluate(), None

        