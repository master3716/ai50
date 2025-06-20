
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



load_data("shopping.csv")