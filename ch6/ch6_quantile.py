import numpy as np
import matplotlib.pyplot as plt

plt.style.use('grayscale')
for alpha in np.linspace(0.1, 0.9, 5):
    x = np.array(range(-10, 11))
    y = np.where(x > 0, alpha * x, (alpha - 1) * x)
    plt.plot(x, y, label=f'Q = {alpha}')
plt.legend(loc='upper left')
plt.savefig('quantile.png')
plt.show()


x = np.linspace(0, 120, 200)
y = np.abs(x)
alpha = 0.5
y_Q = np.where(x > 0, alpha * x, (alpha - 1) * x)
y_logcosh = np.log(np.cosh(x))*0 + np.log(np.exp(x)-np.exp(100))
plt.plot(x, y, label=f'MAE Objective')
plt.plot(x, y_Q, label=f'Q = 0.5')
plt.plot(x, y_logcosh, label=f'Smooth MAE : log_cosh')
plt.legend(loc='upper left')
plt.show()


x = np.linspace(-20, 20, 200)
y = np.abs(x)
alpha = 0.2
y_Q = np.where(x > 0, alpha * x, (alpha - 1) * x)
y_logcosh = np.where(x > 0, alpha * np.log(np.cosh(x)), (1 - alpha) * np.log(np.cosh(x)))
plt.plot(x, y_Q, label=f'Q = 0.2')
plt.plot(x, y_logcosh, label=f'Smooth Q=0.2 regression avec log_cosh')
plt.legend(loc='upper left')
plt.savefig('quantile_02.png')
plt.show()
