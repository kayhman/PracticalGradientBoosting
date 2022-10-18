import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

t = np.array([0, 1])
v = [1, 10]
p = [2, 9]
e = [2, 1]


plt.style.use('grayscale')
fig = plt.figure()
ax1 = fig.add_subplot()
mape = (1 + 1.0/10) / 2 * 100
width = 0.1

ax1.plot(t, [mape, mape], label='MAPE')
ax1.tick_params(axis='y')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

ax2 = ax1.twinx()

ax2.bar(t - width/2, v, width=width, label='Valeur réelle')
ax2.bar(t + width/2, p, width=width, label='Prédiction')


fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.legend()
plt.savefig('MAPE.png')
plt.show()


def MAPE(y_true, y_pred):
    err = y_true - y_pred
    mape = np.mean(abs(err) / y_true)
    return mape

y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 2.7, 4.1])
print(MAPE(y_true, y_pred))
# -> 0. 0687
