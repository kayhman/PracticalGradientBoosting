# hp_xgb_gamma.py
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
print(pred) # -> [-1.1452967  -0.9385602  -0.82778156  0.9230837   1.0179658   1.2299678  1.0467842 ]

plot_tree(model, num_trees=0)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('gamma_overfit.png')
fig.show()

# regularisation
model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=3,
                     gamma=1,
                     reg_alpha=0,
                     reg_lambda=0)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [-0.9705461 -0.9705461 -0.9705461  1.0544504  1.0544504  1.0544504  1.0544504]

plot_tree(model, num_trees=0)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('gamma_regularize.png')
fig.show()
