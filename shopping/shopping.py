import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    month_translator = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }
    evidence = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            line_list = line.split(",")
            try:
                line_list[0] = int(line_list[0])
                line_list[1] = float(line_list[1])
                line_list[2] = int(line_list[2])
                line_list[3] = float(line_list[3])
                line_list[4] = int(line_list[4])
                line_list[5] = float(line_list[5])
                line_list[6] = float(line_list[6])
                line_list[7] = float(line_list[7])
                line_list[8] = float(line_list[8])
                line_list[9] = float(line_list[9])
                line_list[10] = month_translator[line_list[10]]
                line_list[11] = int(line_list[11])
                line_list[12] = int(line_list[12])
                line_list[13] = int(line_list[13])
                line_list[14] = int(line_list[14])
                line_list[15] = 1 if line_list[15].lower() == "Returning_Visitor".lower() else 0
                line_list[16] = bool(line_list[16])
                label = line_list[17]
                line_list.remove(line_list[17])
                evidence.append(line_list)
                labels.append(label)
            except:
                continue
            
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model




def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity_specificity = [0, 0]
    for actual, predicted in zip(labels, predictions):
        if actual == predicted and actual == 1:
            sensitivity_specificity[0] += 1
        elif actual == predicted and actual == 0:
            sensitivity_specificity[1] += 1
    sensitivity_specificity[0] /= len(labels) * 0.5
    sensitivity_specificity[1] /= len(labels) * 0.5
    return tuple(sensitivity_specificity)
    


if __name__ == "__main__":
    main()
