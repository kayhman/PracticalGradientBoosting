# ch6_softmax.py
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from xgboost import plot_tree

x_train, y_train = make_classification(n_samples=100,
                                       n_informative=5,
                                       n_classes=3,
                                       random_state=5)

model = XGBClassifier()

model.fit(x_train, y_train)

#plt.style.use('grayscale')
#plot_tree(model, num_trees=0)
#plt.show()


dump_list = model.get_booster().get_dump()
num_trees = len(dump_list)
print(num_trees)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)
