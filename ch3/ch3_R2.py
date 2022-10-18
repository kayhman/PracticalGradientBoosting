import numpy as np
import math

def R2(y_true, y_pred):
    err = y_true - y_pred
    y_true_mean = np.mean(y_true)
    mean_err = y_true - y_true_mean
    r2 = 1.0 - np.sum(err.dot(err)) / np.sum(mean_err.dot(mean_err))
    return r2

y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 2.7, 4.1])
y_mean = np.array([2.5] * 4)
print(R2(y_true, y_pred))
# -> 0.976
print(R2(y_true, y_mean))
# -> 0.346
