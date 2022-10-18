import numpy as np
import matplotlib
import matplotlib.pyplot as plt


t = np.linspace(-2.0, 2.0, 101)
s = t**2
mae = np.where(t < 0, -t, t)

plt.style.use('grayscale')
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(t, s, label='MSE')
ax.plot(t, mae, ':', label='MAE')
#ax.xlabel('err')
ax.legend()
plt.savefig('MSE.png')
plt.show()


def MSE(y_true, y_pred):
    err = y_true - y_pred
    mse = np.mean(err.dot(err))
    return mse

y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 2.7, 4.1])
print(MSE(y_true, y_pred))
# -> 0.119
