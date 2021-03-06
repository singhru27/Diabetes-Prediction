import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def get_data_as_list(filepath):
    """
    Returns data from the given filepath in list form
    Parameters:
        - filepath: A path to the CSV file we are looking to analyze
    Returns:
        - data_list: Our CSV data in list form
    """
    # Temporary holder for the data
    data_list = []
    with open(filepath, encoding="utf-8-sig") as csvfile:
        datareader = csv.reader(csvfile, delimiter=",")
        for row in datareader:
            data_list.append(row)
    return data_list


def convert_to_numpy_array(data_list):
    """
    Takes in data in list format, and returns the data processed into numpy arrays
    Parameters:
        - data_list: A list of data items
    Returns:
        - (classification_labels, data_array): A numpy array of classifications and the dataset with the labels remove
    """
    # Retrieving a list of indices corresponding to attribute values
    header = data_list[0]
    attribute_to_col = {}
    for idx, item in enumerate(header):
        attribute_to_col[item] = idx
    # Converting the data_list to a numpy array
    data_array = np.array(data_list[1:], dtype=np.float32)
    time_in_hospital = data_array[:, attribute_to_col["time_in_hospital"]]
    # Removing unneeded columns from the dataset
    col_to_remove = []
    col_to_remove.append(attribute_to_col["ID"])
    col_to_remove.append(attribute_to_col["time_in_hospital"])
    data_array = np.delete(data_array, col_to_remove, axis=1)
    # Returning the time in hospital labels and the data_array
    return time_in_hospital, data_array


def train_test_split(data_array, data_labels):
    """
    Splits the data into a training set and a testing set
    Parameters:
        - data_array: Array of the raw data values
        - data_labels: Array of the data labels
    Returns:
        - train_array: Array of the raw data values for the training set
        - train_labels: Array of the labels for the training set
        - test_array: Array of the raw data values for the testing set
        - test_labels: Array of the raw data labels for the testing set
    """
    train_proportion = 0.7
    test_proportion = 0.3
    # Dataset descriptor variables
    num_elements = data_labels.shape[0]
    num_in_train = int(train_proportion * num_elements)
    # Shuffling the data_array and data_labels in the same manner
    shuffled_indices = np.random.permutation(data_array.shape[0])
    data_array = data_array[shuffled_indices]
    data_labels = data_labels[shuffled_indices]
    # Getting train and test datasets
    train_data = data_array[0:num_in_train]
    test_data = data_array[num_in_train:]
    train_labels = data_labels[0:num_in_train]
    test_labels = data_labels[num_in_train:]
    return train_data, train_labels, test_data, test_labels


def main():
    data_list = get_data_as_list("../Data/Cleaned Data/one_hot_cleaned_data.csv")
    data_labels, data_array = convert_to_numpy_array(data_list)
    # Retrieving train and test data sets
    train_data, train_labels, test_data, test_labels = train_test_split(
        data_array, data_labels
    )
    depth = []
    accuracy = []
    for i in range(1, 70, 5):
        print(i)
        classifier = RandomForestRegressor(
            max_depth=i, max_features=100, max_samples=20000
        )
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        mse = mean_squared_error(predictions, test_labels)
        accuracy.append(mse)
        depth.append(i)
    plt.plot(depth, accuracy)
    plt.title("Accuracy Analysis as a Function of Random Forest Depth")
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()


if __name__ == "__main__":
    main()