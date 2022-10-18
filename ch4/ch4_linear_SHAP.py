# ch4_linear_SHAP.py
from itertools import permutations
import math

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def list_all_possible_perumations(nb_features):
    features = set(range(0, nb_features))
    return [cmb for cmb in permutations(features)]


def compute_theta_i(model, nb_features, data, feature_excluded):
    ordering = list_all_possible_perumations(nb_features)
    theta = 0.0
    for subset in ordering:
        subset = list(subset)
        feature_idx = subset.index(feature_excluded)
        data_without_feature = [[x if subset.index(i) < feature_idx else 0 for i, x in enumerate(data)]]
        data_with_feature =  [[x if subset.index(i) <= feature_idx else 0 for i, x in enumerate(data)]]
        weight = 1.0 / len(ordering)
        theta += weight * (model.predict(data_with_feature)[0] - model.predict(data_without_feature)[0])
    return theta


def train_linear_model(n_features):
    x_train, y_train = make_regression(n_samples=100,
                                       n_features=n_features,
                                       n_informative=n_features,
                                       noise=1,
                                       random_state=42)

    model = LinearRegression(fit_intercept=True)
    model.fit(x_train, y_train)
    return x_train, y_train, model


n_features = 2
x_train, y_train, model = train_linear_model(n_features)
theta = []
for feature_idx in range(0,n_features):
    theta.append(compute_theta_i(model, n_features, list(x_train[0]), feature_idx))
print(theta[0], model.coef_[0] * x_train[0][0])
# -> -104.31510227037786 -104.31510227037786
print(theta[1], model.coef_[1] * x_train[0][1])
# -> 48.63805006076018 48.63805006076018
print(np.sum(theta) + model.intercept_, model.predict(x_train[[0]])[0])
# -> -55.655416401171586 -55.655416401171586

n_features = 3
x_train, y_train, model = train_linear_model(n_features)
theta = []
for feature_idx in range(0,n_features):
    theta.append(compute_theta_i(model, n_features, list(x_train[0]), feature_idx))
print(theta[0], model.coef_[0] * x_train[0][0])
# -> -22.36084079197455 -22.36084079197455
print(theta[1], model.coef_[1] * x_train[0][1])
# -> 37.85105905959119 37.85105905959119
print(theta[2], model.coef_[2] * x_train[0][2])
# -> -2.0479756083176737 -2.0479756083176737
print(np.sum(theta) + model.intercept_, model.predict(x_train[[0]])[0])
# -> 13.567167816890098 13.567167816890098


exit(0)


plt.scatter(x_train[:, [0]], y_train)
plt.scatter(x_train[:, [1]], y_train)
#plt.show()

model2 = LinearRegression(fit_intercept=True)
model2.fit(x_train, y_train)

model0 = LinearRegression(fit_intercept=True)
model0.fit(x_train[:, [0]], y_train)
print(model0.coef_)

model1 = LinearRegression(fit_intercept=True)
model1.fit(x_train[:, [1]], y_train)
print(model1.coef_)

# Analysing feature 1
x = x_train[[0]]
x_0 = np.array([[x_train[:, 0][0], 0]])
x_1 = np.array([[0, x_train[:, 1][0]]])
print('x', x)
print(x_0)
print(x_1)
print('y', model2.predict(x), y_train[0])

theta_0 = model2.predict(x) - model2.predict(x_0)
theta_1 = model2.predict(x) - model2.predict(x_1)

#theta_0 = model2.coef_[0] * (x_0[0] - np.mean(x_train[:, 0]) * 0)
#theta_1 = model2.coef_[1] * (x_1[0] - np.mean(x_train[:, 1]) * 0)
print(theta_0)
print(theta_1)
print(model2.intercept_)
print(theta_0 + theta_1 + model2.intercept_)
