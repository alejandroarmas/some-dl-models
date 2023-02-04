[Data Source](https://drive.google.com/file/d/1WG0NV7JTcDmpcEB-GRf-qecyWTisFYM-/view)

### Dataset size:

The dataset has two files: train.csv and test.csv. They denote the pre-partitioned training set and testing set, respectively. The train.csv file has 60,000 lines, and the test.csv file has 10,000 lines. Each line represents one labeled data instance in these two files.

### Data Instance Format:

For each line in the files, there are 785 elements separated by comma.
label,feature1,feature2,feature3,...,feature784

The first element denotes the label with value from {0, 1, 2, …, 9}, and the remaining 784 elements are the features of the data instance. Both the label and features are in integers.

### Task To Be Done:

Please learn a MLP model based on the training set, and evaluate its performance on the testing set and report its learning performance.
