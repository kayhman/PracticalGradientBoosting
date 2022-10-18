# ch6_log_cosh.py
import math
import numpy as np
import matplotlib.pyplot as plt

def logcosh(err):
    loss = math.log(math.cosh(err))
    return loss

t = np.linspace(-10.0, 10.0, 101)
s = np.vectorize(logcosh)(t)
mae = np.where(t > 0, t, -t)


plt.style.use('grayscale')
plt.plot(t, s, label='log(cosh(x))')
plt.plot(t, mae, label='MAE')
plt.xlabel('erreur')
plt.legend()
plt.savefig('log_cosh.png')
plt.show()


def log_cosh(y_true, y_pred):
    err = y_pred - y_true
    grad = np.tanh(err)
    hess = 1 / np.cosh(err)**2
    return grad, hess
