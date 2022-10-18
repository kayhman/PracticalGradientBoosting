from xgboost import XGBRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create time serie timestamp indices
monthly_variation = [0.0, 0.0, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.4, 0.2, 0.05, 0.0]
weekly_sales = [20, 25, 30, 25, 20, 30, 0]
constant_sales = [10, 10, 10, 10, 10, 10, 10]


print(len( pd.date_range(start='01/04/2021', end='01/02/2022')))
X = pd.DataFrame({'date': pd.date_range(start='01/04/2021', end='01/02/2022'),
                  'value': np.array(weekly_sales * 52) + np.random.normal(0, 2, 364)})

X['day_of_week'] = X['date'].apply(lambda date: date.day_of_week)
X['month'] = X['date'].apply(lambda date: date.month)

train_data = lgb.Dataset(X[['day_of_week', 'month']], label=X['value'])
model = lgb.train({}, train_data)

test = pd.DataFrame({'day_of_week': [0, 1, 2, 3, 4, 5, 6],
                     'month': [1, 1, 1, 1, 1, 1, 1]})

pred = model.predict(test)


print(pred)
# -> [20.1 25.02 29e.56 24.99 19.47 30.16 1.4e-02]
print(weekly_sales)
# -> [20, 25, 30, 25, 20, 30, 0]
print(pred - np.array(weekly_sales))
# -> [0.12  0.02 -0.43 -0.00 -0.52  0.16 0.01]

plt.style.use('grayscale')
lgb.plot_importance(model, importance_type='gain')
plt.savefig('ts_gain.png')
plt.show()
