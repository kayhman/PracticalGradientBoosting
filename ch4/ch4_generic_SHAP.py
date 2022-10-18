# ch4_generic_SHAP.py
from itertools import permutations
import math

import numpy as np
import xgboost as xgb
import shap
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def list_all_possible_perumations(nb_features):
    features = set(range(0, nb_features))
    return [cmb for cmb in permutations(features)]

class ZeroModel():
    def predict(self, dummy):
        return [0.0]

def train_model(x_train, y_train):
    # model with empty feature set return 0 by cosntruction
    if x_train.shape[1] == 0:
        return ZeroModel()

    model = xgb.XGBRegressor(n_estimators=5,
                             max_depth=3)
    model.fit(x_train, y_train)
    return model


def compute_theta_i(x_train, y_train,
                    data,
                    feature_excluded):
    nb_features =x_train.shape[1]
    ordering = list_all_possible_perumations(nb_features)
    theta = 0.0
    for subset in ordering:
        subset = list(subset)
        feature_idx = subset.index(feature_excluded)

        subset_with_feature = sorted([fidx for _, fidx in enumerate(subset) if subset.index(fidx) <= feature_idx])
        subset_without_feature = sorted([fidx for _, fidx in enumerate(subset) if subset.index(fidx) < feature_idx])

        weight = 1.0 / len(ordering)
        model_with_feature = train_model(x_train[:, subset_with_feature], y_train)
        model_without_feature = train_model(x_train[:, subset_without_feature], y_train)

        data_with_feature = np.array([[x for i, x in enumerate(data) if i in subset_with_feature]])
        data_without_feature = np.array([[x for i, x in enumerate(data) if i in subset_without_feature]])

        theta += weight * (model_with_feature.predict(data_with_feature)[0] - \
                           model_without_feature.predict(data_without_feature)[0])
    return theta


n_features = 2
x_train, y_train = make_regression(n_samples=100,
                                   n_features=n_features,
                                   n_informative=n_features,
                                   noise=1,
                                   random_state=42)

theta = []
model_full = train_model(x_train, y_train)
explainer = shap.Explainer(model_full)
shap_values = explainer(x_train)

for feature_idx in range(0,n_features):
    theta.append(compute_theta_i(x_train, y_train,
                                 list(x_train[0]), feature_idx))
    print(f'fi_{feature_idx}', theta[feature_idx], shap_values[0].values[feature_idx])
# -> fi_0 -68.42814254760742 -82.14687
# -> fi_1 21.016868591308594 41.031513

print(np.sum(theta) + (model_full.base_score or 0), model_full.predict(x_train[[0]])[0])
# -> -47.41127395629883 -47.411274

n_features = 3
x_train, y_train = make_regression(n_samples=100,
                                   n_features=n_features,
                                   n_informative=n_features,
                                   noise=1,
                                   random_state=42)

theta = []
model_full = train_model(x_train, y_train)
explainer = shap.Explainer(model_full)
shap_values = explainer(x_train)
#print(shap_values[0], y_train.mean())
for feature_idx in range(0,n_features):
    theta.append(compute_theta_i(x_train, y_train,
                                 list(x_train[0]), feature_idx))
    print(f'fi_{feature_idx}', theta[feature_idx], shap_values[0].values[feature_idx])
# -> fi_0 -5.386728286743163 -5.900916
# -> fi_1 11.116163094838459 11.959647
# -> fi_2 2.7234941323598223 -1.0982516

print(np.sum(theta) + (model_full.base_score or 0), model_full.predict(x_train[[0]])[0])
# -> 8.452928940455118 8.452929
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
