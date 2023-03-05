from code.base_class.dataset import datasetConfig
from code.stage_4_code.Dataset_Loader_RNN_Generator import Dataset_Loader_RNN

def main():

    d_config = datasetConfig(
            {
                "name": "data",
                "description": "The dataset contains 1622 pieces of short jokes.",
                "source_folder_path": "data/stage_4_data/",
                "source_file_name": "data.csv",
            }
        )
    
    d = Dataset_Loader_RNN(d_config)
    d.load(clean=1)


if __name__ == "__main__": 
    main()
