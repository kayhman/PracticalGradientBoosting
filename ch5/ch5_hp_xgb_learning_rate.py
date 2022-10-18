# hp_xgb_learning_rate.py
import pandas as pd
from xgboost import XGBRegressor

from xgboost import plot_tree
import matplotlib.pyplot as plt
import matplotlib

x_train = pd.DataFrame({"A" : [3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]})
y_train = pd.DataFrame({"Y" : [3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]})

model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=3,
                     gamma=0,
                     reg_alpha=0,
                     reg_lambda=0)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [3. 2. 1. 4. 5. 6. 7.]

plt.style.use('grayscale')
plot_tree(model, num_trees=0)
plt.savefig('lr_1.png')
plt.show()


model = XGBRegressor(n_estimators=1,
                     learning_rate=0.8,
                     base_score=0,
                     max_depth=3,
                     gamma=0,
                     reg_alpha=0,
                     reg_lambda=0)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [2.4 1.6 0.8 3.2 4.  4.8 5.6]

plt.style.use('grayscale')
plot_tree(model, num_trees=0)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('lr_08.png')
fig.show()
