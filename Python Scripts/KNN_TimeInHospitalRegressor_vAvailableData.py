import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
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
    time_in_hospital = data_array[:, attribute_to_col["time_in_hospital"]]
    # Removing unneeded columns from the dataset
    col_to_remove = []
    col_to_remove.append(attribute_to_col["ID"])
    col_to_remove.append(attribute_to_col["time_in_hospital"])
    col_to_remove.append(attribute_to_col["discharge_disposition_1"])
    col_to_remove.append(attribute_to_col["discharge_disposition_2"])
    col_to_remove.append(attribute_to_col["discharge_disposition_3"])
    col_to_remove.append(attribute_to_col["discharge_disposition_4"])
    col_to_remove.append(attribute_to_col["discharge_disposition_5"])
    col_to_remove.append(attribute_to_col["discharge_disposition_6"])
    col_to_remove.append(attribute_to_col["discharge_disposition_7"])
    col_to_remove.append(attribute_to_col["discharge_disposition_8"])
    col_to_remove.append(attribute_to_col["discharge_disposition_9"])
    col_to_remove.append(attribute_to_col["discharge_disposition_10"])
    col_to_remove.append(attribute_to_col["discharge_disposition_11"])
    col_to_remove.append(attribute_to_col["discharge_disposition_12"])
    col_to_remove.append(attribute_to_col["discharge_disposition_13"])
    col_to_remove.append(attribute_to_col["discharge_disposition_14"])
    col_to_remove.append(attribute_to_col["discharge_disposition_15"])
    col_to_remove.append(attribute_to_col["discharge_disposition_16"])
    col_to_remove.append(attribute_to_col["discharge_disposition_17"])
    col_to_remove.append(attribute_to_col["discharge_disposition_18"])
    col_to_remove.append(attribute_to_col["discharge_disposition_19"])
    col_to_remove.append(attribute_to_col["discharge_disposition_20"])
    col_to_remove.append(attribute_to_col["discharge_disposition_21"])
    col_to_remove.append(attribute_to_col["discharge_disposition_22"])
    col_to_remove.append(attribute_to_col["discharge_disposition_23"])
    col_to_remove.append(attribute_to_col["discharge_disposition_24"])
    col_to_remove.append(attribute_to_col["discharge_disposition_25"])
    col_to_remove.append(attribute_to_col["discharge_disposition_26"])
    col_to_remove.append(attribute_to_col["discharge_disposition_27"])
    col_to_remove.append(attribute_to_col["discharge_disposition_28"])
    col_to_remove.append(attribute_to_col["num_procedures"])
    col_to_remove.append(attribute_to_col["num_lab_procedures"])
    col_to_remove.append(attribute_to_col["num_medications"])
    col_to_remove.append(attribute_to_col["payer_code_MC"])
    col_to_remove.append(attribute_to_col["payer_code_MD"])
    col_to_remove.append(attribute_to_col["payer_code_HM"])
    col_to_remove.append(attribute_to_col["payer_code_UN"])
    col_to_remove.append(attribute_to_col["payer_code_BC"])
    col_to_remove.append(attribute_to_col["payer_code_SP"])
    col_to_remove.append(attribute_to_col["payer_code_CP"])
    col_to_remove.append(attribute_to_col["payer_code_SI"])
    col_to_remove.append(attribute_to_col["payer_code_DM"])
    col_to_remove.append(attribute_to_col["payer_code_CM"])
    col_to_remove.append(attribute_to_col["payer_code_CH"])
    col_to_remove.append(attribute_to_col["payer_code_PO"])
    col_to_remove.append(attribute_to_col["payer_code_WC"])
    col_to_remove.append(attribute_to_col["payer_code_OT"])
    col_to_remove.append(attribute_to_col["payer_code_OG"])
    col_to_remove.append(attribute_to_col["payer_code_MP"])
    col_to_remove.append(attribute_to_col["payer_code_FR"])
    col_to_remove.append(attribute_to_col["readmitted"])

    data_array = np.delete(data_array, col_to_remove, axis=1)
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
    print(data_array.shape)
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
            # Getting the K-Neighbors regressor and classifying
            regressor = KNeighborsRegressor(n_neighbors=j)
            regressor.fit(train_data, train_labels)
            predictions = regressor.predict(test_data)
            mse = mean_squared_error(predictions, test_labels)
            if i == 5:
                PCA_5.append(mse)
            if i == 10:
                PCA_10.append(mse)
            if i == 15:
                PCA_15.append(mse)
            if i == 20:
                PCA_20.append(mse)
    # Plotting all 5 graphs together
    plt.plot(k_values, PCA_5, label="5 Components")
    plt.plot(k_values, PCA_10, label="10 Components")
    plt.plot(k_values, PCA_15, label="15 Components")
    plt.plot(k_values, PCA_20, label="20 Components")
    plt.title("Accuracy Analysis")
    plt.xlabel("k-values")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()