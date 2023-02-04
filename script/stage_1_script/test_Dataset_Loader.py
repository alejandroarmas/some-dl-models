# this test script verifies the data is being loaded and extracted correctly in Data_Loader.py 
# (need to add test.csv and train.csv to data/stage_2_data locally to run)

# ToDo: update Data_Loader.py to remove hard coded values; 
# see if we can collect data from google drive instead of having to add it to repo

# use 'poetry run python -m script.stage_1_script.test_Dataset_Loader' while in project folder to run this program

from code.stage_1_code.Dataset_Loader import Dataset_Loader

# initialize dataset object 
config = {'name': 'stage_2_data_test', 'description': '...', 'source_folder_path': './data/stage_2_data/', 'source_file_name': 'test.csv'}
data_obj = Dataset_Loader(config)

X, y = data_obj.load()
print(X, "\n")
print(y, "\n")
