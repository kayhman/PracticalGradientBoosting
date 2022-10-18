import numpy as np
import math

def RMSE(y_true, y_pred):
    err = y_true - y_pred
    rmse = math.sqrt(np.mean(err.dot(err)))
    return rmse

y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 2.7, 4.1])
print(RMSE(y_true, y_pred))
# -> 0.346
