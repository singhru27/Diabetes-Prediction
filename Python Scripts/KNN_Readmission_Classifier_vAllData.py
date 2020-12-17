import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    # Only adding selected variables to the dataset
    # Returning the classification labels and the data_array
    return readmitted_labels, data_array


def KNN(train_data, train_labels, test_data, k):
    """
    Classified each point in train_data according to the KNN algorithm
    Parameters:
        - train_data: A numpy array, where each row represents a datapoint in our "training" set
        - train_labels: A numpy array, where each element represents the label for readmission
        - test_data: A numpy array, where each row represents a datapoint that must be classified
        - k: A hyperparameter that determines how many of the closest neighbors to examine
    Returns:
        - predicted_labels: A vector of predictions for the test_data
    """
    # Looping through each element in the array, getting the distances and classifications for each
    test_labels = np.zeros(test_data.shape[0])
    for i in range(test_data.shape[0]):
        dist_data = np.linalg.norm(train_data - test_data[i], axis=1)
        idx = np.argpartition(dist_data, k)[:k]
        # Retrieving the labels
        k_labels = train_labels[idx]
        # Getting the most common label
        (values, counts) = np.unique(k_labels, return_counts=True)
        ind = np.argmax(counts)
        label = values[ind]
        test_labels[i] = int(label)
    return test_labels


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
    train_proportion = 0.8
    test_proportion = 0.2
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


def get_accuracy(predicted_labels, true_labels):
    num_elements = len(predicted_labels)
    correct_labels = predicted_labels == true_labels
    return np.sum(correct_labels) / num_elements


def main():
    data_list = get_data_as_list("../Data/Cleaned Data/one_hot_cleaned_data.csv")
    data_labels, data_array = convert_to_numpy_array(data_list)
    # Normalizing the data for usage in the KNN algorithm
    scaler = StandardScaler()
    data_array_normalized = scaler.fit_transform(data_array)
    # List holders for accuracy and k values
    k_values = [1, 4, 7, 10, 13, 16, 19, 22]
    PCA_5 = []
    PCA_10 = []
    PCA_15 = []
    PCA_20 = []
    for i in range(5, 25, 5):
        # Dimensionality reduction
        pca = PCA(n_components=i)
        data_array_transformed = pca.fit_transform(X=data_array_normalized)
        # Getting the training and testing data
        train_data, train_labels, test_data, test_labels = train_test_split(
            data_array_transformed, data_labels
        )
        for j in range(1, 25, 3):
            print(j)
            predicted_labels = KNN(train_data, train_labels, test_data, k=j)
            if i == 5:
                PCA_5.append(get_accuracy(predicted_labels, test_labels))
            if i == 10:
                PCA_10.append(get_accuracy(predicted_labels, test_labels))
            if i == 15:
                PCA_15.append(get_accuracy(predicted_labels, test_labels))
            if i == 20:
                PCA_20.append(get_accuracy(predicted_labels, test_labels))
    # Plotting all 5 graphs together
    plt.plot(k_values, PCA_5, label="5 Components")
    plt.plot(k_values, PCA_10, label="10 Components")
    plt.plot(k_values, PCA_15, label="15 Components")
    plt.plot(k_values, PCA_20, label="20 Components")
    plt.title("Accuracy Analysis")
    plt.xlabel("k-values")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()