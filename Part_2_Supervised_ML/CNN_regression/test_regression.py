import pandas as pd
from data_generator import DataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix


def calculate_accuracy(outcome, labels):
    outcome_cat = []
    for value in outcome:
        if value.item() < - 0.2:
            outcome_cat.append(1)
            continue
        if value.item() > 0.2:
            outcome_cat.append(2)
            continue
        else:
            outcome_cat.append(0)

    labels_cat = []
    for value in labels:
        if value < - 0.2:
            labels_cat.append(1)
            continue
        if value > 0.2:
            labels_cat.append(2)
            continue
        else:
            labels_cat.append(0)
    print("outcome_cat:", outcome_cat)
    print("labels_cat:", labels_cat)

    accuracy = accuracy_score(outcome_cat, labels_cat)
    return accuracy