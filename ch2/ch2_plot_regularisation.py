# plot_regularisation.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


x_train = pd.DataFrame({"A" : [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]})
y_train = pd.DataFrame({"Y" : [-1, -1, -1, 1.0, 1.0, 1.0, 1.0]})
y_train['Y'] = y_train['Y'] + np.random.normal(0, .1, y_train.shape[0])

plt.style.use('grayscale')
plt.plot(x_train['A'],
         y_train['Y'],
         'o',
         label="Données d'entrainement")
plt.plot(x_train['A'],
         [-1, -1, -1, 1.0, 1.0, 1.0, 1.0],
         '-',
        label='Valeurs théorique à prédire')
plt.plot(x_train['A'],
         y_train['Y'],
         '+',
        label='Prédictions du modèle sur-appris')
plt.plot(x_train['A'],
         [-0.99, -0.99, -0.99,  1.06,   1.06,   1.06, 1.06],
         '-.',
        label='Prédictions du modèle régularisé')


plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
plt.savefig('regularisation.png')

plt.clf()
plt.style.use('grayscale')
plt.plot(x_train['A'],
         y_train['Y'],
         'o',
         label="Données d'entrainement")
plt.plot(x_train['A'],
         [-1, -1, -1, 1.0, 1.0, 1.0, 1.0],
         '-',
        label='Valeurs théorique à prédire')
plt.plot(x_train['A'],
         [-1.0416804492739518, -1.0616216178137203, -1.02938621165474, 0.9898047539926471, 1.1940988659259375, 0.9751023136016387, 0.9891703732728606],
         '-.',
        label='Généralisation avec le modèle sur appris')
plt.plot(x_train['A'],
         [-0.99, -0.99, -0.99,  1.06,   1.06,   1.06, 1.06],
         ':',
        label='Généralisation avec le modèle régularisé')

plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
plt.savefig('generalisation.png')
plt.show()
