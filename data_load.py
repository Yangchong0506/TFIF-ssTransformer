"""
Data loading and evaluation utilities for TFIF-ssTransformer.

Provides metric calculation and dataset statistics loading for the
semi-supervised FeO soft sensing training pipeline.
"""

import numpy as np
import pandas as pd
import math


def score_pre(y_test, y_pre, delta_limit):
    """Calculate regression metrics: MSE, RMSE, R², hit rate, MAPE."""
    mse = np.sum((y_test - y_pre) ** 2) / len(y_test)
    rmse = np.sqrt(mse)
    var = np.var(y_test)
    r2 = 1 - mse / var
    mape = np.mean(np.abs((y_test - y_pre) / y_test)) * 100
    delta = np.abs(y_test - y_pre).reshape(1, -1)
    right = np.count_nonzero(delta < delta_limit)
    hr = right / len(y_test)
    return mse, rmse, r2, hr, mape


def get_average(data):
    return sum(data) / len(data)


def get_variance(data):
    average = get_average(data)
    return sum([(x - average) ** 2 for x in data]) / len(data)


def get_standard_deviation(data):
    variance = get_variance(data)
    return math.sqrt(variance)


def get_ss_did3_data_original():
    """
    Load labelled FeO values and compute z-score normalization statistics.

    Returns:
        train_y_org: training set FeO labels (original scale)
        test_y_org:  test set FeO labels (original scale)
        avg_train_y_org: mean of training labels (for denormalization)
        std_train_y_org: std of training labels (for denormalization)
    """
    data_all = pd.read_csv('data/ss_did3_data_all.csv')

    data_y = data_all.iloc[:, -1:]
    data_y_select = data_y[data_y.index % 4 == 0]
    train_y_num = int(data_y_select.shape[0] * 0.7)
    train_y_org = np.array(data_y_select[:train_y_num])
    test_y_org = np.array(data_y_select[train_y_num:])
    avg_train_y_org = get_average(train_y_org)
    std_train_y_org = get_standard_deviation(train_y_org)
    return train_y_org, test_y_org, avg_train_y_org, std_train_y_org
