# ch6_squared.py
import numpy as np
import matplotlib.pyplot as plt

def squared_error(x):
    return x**2

t = np.linspace(-10.0, 10.0, 101)
s = np.vectorize(squared_error)(t)

plt.style.use('grayscale')
plt.plot(t, s, label='Carré de l\'erreur')
plt.xlabel('erreur')
plt.ylabel('se(y_true, y_pred)')
plt.legend()
plt.savefig('squared_error.png')
plt.show()
