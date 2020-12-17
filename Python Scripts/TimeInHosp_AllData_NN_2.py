import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf


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
    # Only adding selected variables to the dataset
    # Returning the classification labels and the data_array
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





def main():
    print("main called")
    data_list = get_data_as_list("./Cleaned Data/one_hot_cleaned_data.csv")
    data_labels, data_array = convert_to_numpy_array(data_list)
    
    # Normalizing the data for usage in the KNN algorithm
    print("normalizing data")
    scaler = StandardScaler()
    data_array_normalized = scaler.fit_transform(data_array)
    

    # # Getting the training and testing data
    train_data, train_labels, test_data, test_labels = train_test_split(data_array_normalized, data_labels)
    


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3000, activation='relu'),
        tf.keras.layers.Dropout(.2),
        
        tf.keras.layers.Dense(1500, activation='relu'),
        tf.keras.layers.Dropout(.2),
        
        tf.keras.layers.Dense(800, activation='relu'),
        tf.keras.layers.Dropout(.2),
        
        tf.keras.layers.Dense(400, activation='relu'),

        tf.keras.layers.Dense(1, activation='relu'),
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_squared_error'])

    print("now fitting")
    history = model.fit(train_data, train_labels, batch_size=64, epochs=4)

    print("now evaluating")
    results = model.evaluate(test_data, test_labels)

    print("results here:", results)    
    print(model.summary())



if __name__ == "__main__":
    main()