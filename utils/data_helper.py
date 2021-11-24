'''
Author : Mingu Kang
Date   : Nov 2021
'''
import heapq
import shutil
from skmultilearn.model_selection import IterativeStratification
import torch
import pandas as pd
import numpy as np


def load_data(data_path: pd.DataFrame) -> np.array:
    data = pd.read_csv(data_path)
    X, y = data.iloc[:, 0].to_numpy(), data.iloc[:,1:].to_numpy()
    return X, y

def iterative_train_test_split(X: np.array, y: np.array, split_size: float) -> np.array:
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-split_size, split_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def get_onehot_label_threshold(output, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    output = np.ndarray.tolist(output)
    for score in output:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels

def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted onehot labels based on the topK number.
    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def save_checkpoint(state, filename):
    torch.save(state, filename)
    # shutil.copyfile(filename, 'model_best.pth')