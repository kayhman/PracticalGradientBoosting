import numpy as np
import matplotlib
import matplotlib.pyplot as plt


t = np.linspace(-1.0, 1.0, 101)
s = np.where(t < 0, -t, t)

plt.style.use('grayscale')
plt.plot(t, s, label='MAE')
plt.xlabel('err')
plt.legend()
plt.savefig('MAE.png')
plt.show()


def MAE(y_true, y_pred):
    err = y_true - y_pred
    mae = np.mean(abs(err))
    return mae

y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 2.7, 4.1])
print(MAE(y_true, y_pred))
# -> 0. 1499
