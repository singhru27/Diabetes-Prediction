import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
import matplotlib.pyplot as plt


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
    readmitted_labels = data_array[:, attribute_to_col["readmitted"]]
    # Removing unneeded columns from the dataset
    col_to_remove = []
    col_to_remove.append(attribute_to_col["ID"])
    col_to_remove.append(attribute_to_col["readmitted"])
    data_array = np.delete(data_array, col_to_remove, axis=1)
    # Returning the classification labels and the data_array
    return readmitted_labels, data_array


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
    # Normalizing the data for usage in the KNN algorithm
    scaler = StandardScaler()
    data_array_normalized = scaler.fit_transform(data_array)
    # Retrieving train and test data sets
    train_data, train_labels, test_data, test_labels = train_test_split(
        data_array_normalized, data_labels
    )
    # Creating C-value array for graphing
    c_values = [10, 1, 0.01, 0.001, 0.0001]
    accuracy = []

    for i in range(len(c_values)):
        print(i)
        # Creating and fitting the classifier
        classifier = LinearSVC(C=c_values[i], dual=False, max_iter=2000)
        classifier.fit(train_data, train_labels)
        predictions = classifier.predict(test_data)
        accuracy.append(accuracy_score(predictions, test_labels))
    plt.plot(c_values, accuracy)
    plt.title("Accuracy Analysis as a Function of Regularization - RBF Kernel")
    plt.xlabel("C values")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()