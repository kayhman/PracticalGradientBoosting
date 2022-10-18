# ch6_pseudo_huber.py
import math
import numpy as np
import matplotlib.pyplot as plt

def pseudo_huber(err):
    delta = 0.99
    loss = delta**2 * (math.sqrt(1 + (err / delta)**2) - 1)
    return loss

t = np.linspace(-10.0, 10.0, 101)
s = np.vectorize(pseudo_huber)(t)
mae = np.where(t > 0, t, -t)


plt.style.use('grayscale')
plt.plot(t, s, label='Pseudo huber')
plt.plot(t, mae, label='MAE')
plt.xlabel('erreur')
plt.legend()
plt.savefig('pseudo_huber.png')
plt.show()
