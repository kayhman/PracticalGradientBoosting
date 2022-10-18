# spline.py
import numpy as np
import math
import random
from numpy.linalg import inv

import matplotlib.pyplot as plt

nbSamples = 30

X = np.matrix([[(x + 1) * 1.0 / nbSamples, 1]
               for x in range(nbSamples)])
Y = np.matrix([math.log(x[0].item(0))
               for x in X]).transpose()


def Iplus(xi, x):
    if x >= xi:
        return x - xi
    else:
        return 0.0


def splinify(xMin, xMax, step, x):
    a = [
        Iplus(xMin + i * step, x)
        for i in range(int((xMax - xMin) / step))
    ]
    a.reverse()
    return a + [1]


Xsplines = np.matrix([
    splinify(0.0, 1.0, 1, x[0].item(0)) for x in X
])
A = inv(Xsplines.transpose() *
        Xsplines) * Xsplines.transpose() * Y
YregLine = np.matrix([[np.dot(x, A).item(0)]
                      for x in Xsplines])

Xsplines = np.matrix([
    splinify(0.0, 1.0, 0.5, x[0].item(0))
    for x in X
])
A = inv(Xsplines.transpose() *
        Xsplines) * Xsplines.transpose() * Y
YregCoarse = np.matrix([[np.dot(x, A).item(0)]
                        for x in Xsplines])

Xsplines = np.matrix([
    splinify(0.0, 1.0, 0.05, x[0].item(0))
    for x in X
])
A = inv(Xsplines.transpose() *
        Xsplines) * Xsplines.transpose() * Y
Yreg = np.matrix([[np.dot(x, A).item(0)]
                  for x in Xsplines])

plt.style.use('grayscale')
plt.plot(np.asarray(X[:, 0]),
         np.asarray(Y),
         '+',
         label='Value to predict(log)')
plt.plot(np.asarray(X[:, 0]),
         np.asarray(YregLine),
         label='Linear regression')
plt.plot(np.asarray(X[:, 0]),
         np.asarray(YregCoarse),
         label='2 splines')
plt.plot(np.asarray(X[:, 0]),
         np.asarray(Yreg),
         label='20 splines')
plt.legend(loc="upper left")
plt.savefig('spline.png')
plt.show()
