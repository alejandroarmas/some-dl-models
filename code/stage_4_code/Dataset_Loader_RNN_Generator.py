"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset, datasetConfig
from code.lib.notifier import DatasetNotifier
from typing import Optional
from nltk.tokenize import word_tokenize

import pandas as pd
import re
import nltk
nltk.download('punkt')


class Dataset_Loader_RNN(dataset):
    data = None

    def __init__(self, config: datasetConfig, manager: Optional[DatasetNotifier] = None):
        super().__init__(config, manager)

    def load(self, clean = 0):
        print("\nloading data...\n")
        filePath = self.dataset_source_folder_path + self.dataset_source_file_name
        filePathClean = self.dataset_source_folder_path + "data_clean.csv"
        
        if clean:
            df = pd.read_csv(filePath)

            # remove IDs and quotations from the dataset 
            df = df.iloc[:, 1] 

            # remove links 
            df[:] = df[:].replace(to_replace=r'http\S+', value='', regex=True)

            # remove sequences which start and end with a parenthesis
            df[:] = df[:].replace(to_replace=r'\([^)]*\)', value='', regex=True)

            # remove sequences which start '&g'
            df[:] = df[:].replace(to_replace=r'&g.*', value='', regex=True)

            # remove any rows which contain a '\r'
            df[:] = df[:].replace(to_replace=r'.*\r.*\n?', value='', regex=True)

            # remove characters
            df[:] = df[:].replace(to_replace = r'[\[\]\(\)\*\.\.\.\$\!\-"\'\']', value='', regex=True)

            # append <EOS> to each line
            #df[:] = df[:] + "<EOS>"
            
            # write to file
            print("\nwriting 'clean' data...\n")
            df.to_csv(filePathClean, index=False, header=False)
            
            # load "clean" data
            file = open(filePathClean, 'rt')
            text = file.read()
            file.close()

            # remove all quotations
            clean_text = re.sub(r'["\']', '', text)

            # split into words
            tokens = word_tokenize(clean_text)

            # convert to lower case
            tokens = [w.lower() for w in tokens] 

            print(tokens[:100])

        return