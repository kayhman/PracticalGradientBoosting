# ch6_logistic_classification.py
import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

t = np.linspace(-10.0, 10.0, 101)
s = np.vectorize(sigmoid)(t)
e = np.vectorize(lambda x: x * x)(s)

plt.style.use('grayscale')
plt.plot(t, s, label='Sigmoid')
plt.plot(t, e, label='Squared Error')
plt.xlabel('prediction')
plt.ylabel('sigmoid(prediction)')
plt.legend()
plt.savefig('sigmoid.png')
plt.show()


t = np.linspace(0.001, 0.999, 101)
s1 = [-math.log(x) for x in t]
s2 = [-math.log(1-x) for x in t]

plt.style.use('grayscale')
plt.plot(t, s1, label='-log(p)')
plt.plot(t, s2, ':', label='-log(1-p)')
plt.xlabel('p')
plt.ylabel('erreur')
plt.legend()
plt.savefig('logloss.png')
plt.show()
