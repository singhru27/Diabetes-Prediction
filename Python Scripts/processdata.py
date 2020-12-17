import csv
import pandas as pd
import random


def load_data(datafile):
    """
    Loads the data from the CSV file into a numpy array
    Parameters:
       - datafile: A CSV file
    Returns:
       - labelDict: A dictionary mapping from label names to column numbers
       - data: A numpy array, where each row represents a specific observation
    """

    # Temporary holder for the data
    data_list = []
    with open(datafile) as csvfile:
        datareader = csv.reader(csvfile, delimiter=",")
        for row in datareader:
            data_list.append(row)
    return data_list


def one_hot_encode(data_list):
    """
    Turns a data_list into a pandas data_frame and one hot encodes
    Parameters:
       - data_list: A list of data
    Returns:
       - one_hot_data: A dataframe with one hot encodings of categorical features
    """
    headers = data_list[0]
    data = data_list[1:]
    df = pd.DataFrame(data, columns=headers)
    # Onehot encoding the "diag_1" category
    df = pd.concat([df, pd.get_dummies(df["diag_1"], prefix="diag_1")], axis=1)
    df.drop(["diag_1"], axis=1, inplace=True)
    # Onehot encoding the "diag_2" category
    df = pd.concat([df, pd.get_dummies(df["diag_2"], prefix="diag_2")], axis=1)
    df.drop(["diag_2"], axis=1, inplace=True)
    # Onehot encoding the "diag_3" category
    df = pd.concat([df, pd.get_dummies(df["diag_3"], prefix="diag_3")], axis=1)
    df.drop(["diag_3"], axis=1, inplace=True)
    # Onehot encoding the "max_glu_serum" category
    df = pd.concat(
        [df, pd.get_dummies(df["max_glu_serum"], prefix="max_glu_serum")], axis=1
    )
    df.drop(["max_glu_serum"], axis=1, inplace=True)
    # Onehot encoding the "A1Cresult"
    df = pd.concat([df, pd.get_dummies(df["A1Cresult"], prefix="A1Cresult")], axis=1)
    df.drop(["A1Cresult"], axis=1, inplace=True)
    # Onehot encoding the "metformin" medication
    df = pd.concat([df, pd.get_dummies(df["metformin"], prefix="metformin")], axis=1)
    df.drop(["metformin"], axis=1, inplace=True)
    # Onehot encoding the "repaglinide" medication
    df = pd.concat(
        [df, pd.get_dummies(df["repaglinide"], prefix="repaglinide")], axis=1
    )
    df.drop(["repaglinide"], axis=1, inplace=True)
    # Onehot encoding the "nateglinide" medication
    df = pd.concat(
        [df, pd.get_dummies(df["nateglinide"], prefix="nateglinide")], axis=1
    )
    df.drop(["nateglinide"], axis=1, inplace=True)
    # Onehot encoding the "chlorpropamide" medication
    df = pd.concat(
        [df, pd.get_dummies(df["chlorpropamide"], prefix="chlorpropamide")], axis=1
    )
    df.drop(["chlorpropamide"], axis=1, inplace=True)
    # Onehot encoding the "glimepiride" medication
    df = pd.concat(
        [df, pd.get_dummies(df["glimepiride"], prefix="glimepiride")], axis=1
    )
    df.drop(["glimepiride"], axis=1, inplace=True)
    # Onehot encoding the "acetohexamide" medication
    df = pd.concat(
        [df, pd.get_dummies(df["acetohexamide"], prefix="acetohexamide")], axis=1
    )
    df.drop(["acetohexamide"], axis=1, inplace=True)
    # Onehot encoding the "glipizide" medication
    df = pd.concat([df, pd.get_dummies(df["glipizide"], prefix="glipizide")], axis=1)
    df.drop(["glipizide"], axis=1, inplace=True)
    # Onehot encoding the "glyburide" medication
    df = pd.concat([df, pd.get_dummies(df["glyburide"], prefix="glyburide")], axis=1)
    df.drop(["glyburide"], axis=1, inplace=True)
    # Onehot encoding the "tolbutamide" medication
    df = pd.concat(
        [df, pd.get_dummies(df["tolbutamide"], prefix="tolbutamide")], axis=1
    )
    df.drop(["tolbutamide"], axis=1, inplace=True)
    # Onehot encoding the "pioglitazone" medication
    df = pd.concat(
        [df, pd.get_dummies(df["pioglitazone"], prefix="pioglitazone")], axis=1
    )
    df.drop(["pioglitazone"], axis=1, inplace=True)
    # Onehot encoding the "rosiglitazone" medication
    df = pd.concat(
        [df, pd.get_dummies(df["rosiglitazone"], prefix="rosiglitazone")], axis=1
    )
    df.drop(["rosiglitazone"], axis=1, inplace=True)
    # Onehot encoding the "acarbose" medication
    df = pd.concat([df, pd.get_dummies(df["acarbose"], prefix="acarbose")], axis=1)
    df.drop(["acarbose"], axis=1, inplace=True)
    # Onehot encoding the "miglitol" medication
    df = pd.concat([df, pd.get_dummies(df["miglitol"], prefix="miglitol")], axis=1)
    df.drop(["miglitol"], axis=1, inplace=True)
    # Onehot encoding the "troglitazone" medication
    df = pd.concat(
        [df, pd.get_dummies(df["troglitazone"], prefix="troglitazone")], axis=1
    )
    df.drop(["troglitazone"], axis=1, inplace=True)
    # Onehot encoding the "tolazamide" medication
    df = pd.concat([df, pd.get_dummies(df["tolazamide"], prefix="tolazamide")], axis=1)
    df.drop(["tolazamide"], axis=1, inplace=True)
    # Onehot encoding the "examide" medication
    df = pd.concat([df, pd.get_dummies(df["examide"], prefix="examide")], axis=1)
    df.drop(["examide"], axis=1, inplace=True)
    # Onehot encoding the "citoglipton" medication
    df = pd.concat(
        [df, pd.get_dummies(df["citoglipton"], prefix="citoglipton")], axis=1
    )
    df.drop(["citoglipton"], axis=1, inplace=True)
    # Onehot encoding the "insulin" category
    df = pd.concat([df, pd.get_dummies(df["insulin"], prefix="insulin")], axis=1)
    df.drop(["insulin"], axis=1, inplace=True)
    # Onehot encoding the "glyburide-metformin" category
    df = pd.concat(
        [df, pd.get_dummies(df["glyburide-metformin"], prefix="glyburide-metformin")],
        axis=1,
    )
    df.drop(["glyburide-metformin"], axis=1, inplace=True)
    # Onehot encoding the "glipizide-metformin" category
    df = pd.concat(
        [df, pd.get_dummies(df["glipizide-metformin"], prefix="glipizide-metformin")],
        axis=1,
    )
    df.drop(["glipizide-metformin"], axis=1, inplace=True)
    # Onehot encoding the "glimepiride-pioglitazone" category
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df["glimepiride-pioglitazone"], prefix="glimepiride-pioglitazone"
            ),
        ],
        axis=1,
    )
    df.drop(["glimepiride-pioglitazone"], axis=1, inplace=True)
    # Onehot encoding the "metformin-rosiglitazone" category
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df["metformin-rosiglitazone"], prefix="metformin-rosiglitazone"
            ),
        ],
        axis=1,
    )
    df.drop(["metformin-rosiglitazone"], axis=1, inplace=True)
    # Onehot encoding the "metformin-pioglitazone" category
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df["metformin-pioglitazone"], prefix="metformin-pioglitazone"
            ),
        ],
        axis=1,
    )
    df.drop(["metformin-pioglitazone"], axis=1, inplace=True)
    # Onehot encoding the "change" category
    df = pd.concat(
        [
            df,
            pd.get_dummies(df["change"], prefix="change"),
        ],
        axis=1,
    )
    df.drop(["change"], axis=1, inplace=True)
    # Onehot encoding the "diabetesMed" category
    df = pd.concat(
        [
            df,
            pd.get_dummies(df["diabetesMed"], prefix="diabetesMed"),
        ],
        axis=1,
    )
    df.drop(["diabetesMed"], axis=1, inplace=True)
    return df


def print_to_CSV(df):
    """
    Prints a supplied dataframe to a CSV file
    Parameters:
        - df: A Pandas dataframe
    Returns:
        - None
    """
    df.to_csv("one_hot_cleaned_data.csv")


def split_csv(csv_file):
    """
    Splits the dataset into a train dataset and a test dataset
    Parameters:
        - csv_file: A CSV file
    Returns:
        - None
    """
    i = 0
    with open(csv_file) as fr:
        with open("diabetes_train.csv", "w") as f1, open(
            "diabetes_test.csv", "w"
        ) as f2:
            for line in fr:
                if i == 0:
                    f1.write(line)
                    f2.write(line)
                    i += 1
                    continue
                f = random.choices([f1, f2], weights=[7, 3])
                f[0].write(line)


def main():
    # data_list = load_data("../Data/cleaned_diabetic_data.csv")
    # df = one_hot_encode(data_list)
    # print_to_CSV(df)

    split_csv("../Data/one_hot_cleaned_data.csv")


if __name__ == "__main__":
    main()
