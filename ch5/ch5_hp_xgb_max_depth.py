# hp_xgb_max_depth.py
import pandas as pd
from xgboost import XGBRegressor

from xgboost import plot_tree
import matplotlib.pyplot as plt

x_train = pd.DataFrame({"A" : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]})
y_train = pd.DataFrame({"Y" : [2.0, 4.0, 6.0, 8.0, 20.0, 24.0, 28.0, 36.0]})

model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=3,
                     gamma=0,
                     reg_alpha=0,
                     reg_lambda=0)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [ 2.  4.  6.  8. 20. 24. 28. 36.]

plt.style.use('grayscale')
plot_tree(model, num_trees=0)
plt.style.use('grayscale')
plt.savefig('max_depth_3.png')
plt.show()

model = XGBRegressor(n_estimators=1,
                     learning_rate=1.,
                     base_score=0,
                     max_depth=4,
                     gamma=0,
                     reg_alpha=0,
                     reg_lambda=0)

model.fit(x_train, y_train['Y'])
pred = model.predict(x_train)
print(pred) # -> [ 2.  4.  6.  8. 20. 24. 28. 36.]

plt.style.use('grayscale')
plot_tree(model, num_trees=0)
plt.style.use('grayscale')
plt.savefig('max_depth_4.png')
plt.show()
