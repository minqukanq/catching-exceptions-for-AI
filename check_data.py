'''
Author : Mingu Kang
Date   : Nov 2021
'''

import os
from collections import Counter
import pandas as pd
from utils import data_helper as helper
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix

ROOT_DIR = os.path.abspath(os.curdir)


if __name__ == '__main__':
    data_file = 'sample.csv'

    X, y = helper.load_data(ROOT_DIR+os.sep+'dataset'+os.sep+data_file)

    train_size = 0.8
    test_size = 0.5
    val_size = 1 - test_size

    # Perform Multi-label stratified sampling
    X_train, X_, y_train, y_ = helper.iterative_train_test_split(X, y, split_size=train_size)
    X_val, X_test, y_val, y_test = helper.iterative_train_test_split(X_, y_, split_size=val_size)

    # Counts
    print(f"train: {len(X_train)} ({len(X_train)/len(X):.2f})\n"
      f"val: {len(X_val)} ({len(X_val)/len(X):.2f})\n"
      f"test: {len(X_test)} ({len(X_test)/len(X):.2f})")

    counts = {}
    counts["train_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
        y_train, order=1) for combination in row)
    counts["val_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
        y_val, order=1) for combination in row)
    counts["test_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
        y_test, order=1) for combination in row)

    # Adjust counts across splits
    for k in counts["val_counts"].keys():
        counts["val_counts"][k] = int(counts["val_counts"][k] * \
            (train_size/val_size))
    for k in counts["test_counts"].keys():
        counts["test_counts"][k] = int(counts["test_counts"][k] * \
            (train_size/test_size))
        

    # View distributions
    print(
        pd.DataFrame({
            "train": counts["train_counts"],
            "val": counts["val_counts"],
            "test": counts["test_counts"]
        }).T.fillna(0)
    )
