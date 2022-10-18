# hp_xgb_lambda.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from xgboost import plot_tree
import matplotlib.pyplot as plt
import matplotlib

x_train = pd.DataFrame({"A" : [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]})
y_train = pd.DataFrame({"Y" : [-1, -1, -1, 1.0, 1.0, 1.0, 1.0]})
y_train['Y'] = y_train['Y'] + np.random.normal(0, .1, y_train.shape[0])

# overfitting
model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=3,
                     gamma=0,
                     reg_alpha=0,
                     reg_lambda=0)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [-0.62654    -0.62654    -0.62654     0.77346283  0.77346283  0.77346283 0.77346283]

plot_tree(model, num_trees=0)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('lambda_overfit.png')
fig.show()

# regularisation
model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=3,
                     gamma=0,
                     reg_alpha=0,
                     reg_lambda=1)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [-0.9705461 -0.9705461 -0.9705461  1.0544504  1.0544504  1.0544504  1.0544504]

plot_tree(model, num_trees=0)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('lambda_regularize.png')
fig.show()

# regularisation many data
x_train = pd.DataFrame({"A" : [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 10})
y_train = pd.DataFrame({"Y" : [-1, -1, -1, 1.0, 1.0, 1.0, 1.0] * 10})
y_train['Y'] = y_train['Y'] + np.random.normal(0, .1, y_train.shape[0])

model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=3,
                     gamma=0,
                     reg_alpha=0,
                     reg_lambda=1)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [-0.97340184 -0.97340184 -0.97340184  0.9854295   0.9854295   0.9854295 0.9854295 ...

plot_tree(model, num_trees=0)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('lambda_regularize_mady_data.png')
fig.show()

# regularisation many data gamma
x_train = pd.DataFrame({"A" : [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 10})
y_train = pd.DataFrame({"Y" : [-1, -1, -1, 1.0, 1.0, 1.0, 1.0] * 10})
y_train['Y'] = y_train['Y'] + np.random.normal(0, .1, y_train.shape[0])

model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=3,
                     gamma=1,
                     reg_alpha=0,
                     reg_lambda=0)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> -0.9960624  -0.9960624  -0.9960624   0.99365574  0.99365574  0.99365574 0.99365574

plot_tree(model, num_trees=0)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('gamma_regularize_many_data.png')
fig.show()
