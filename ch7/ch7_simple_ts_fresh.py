from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create time serie timestamp indices
monthly_variation = [0.0, 0.0, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.4, 0.2, 0.05, 0.0]
weekly_sales = [20, 25, 30, 25, 20, 30, 0]
constant_sales = [10, 10, 10, 10, 10, 10, 10]


product_0 = pd.DataFrame({'date': pd.date_range(start='01/04/2021', end='01/02/2022'),
                  'value': np.array(weekly_sales * 52) + np.random.normal(0, 2, 364),
                  'product_id': [0] * 364})
product_1 = pd.DataFrame({'date': pd.date_range(start='01/04/2021', end='01/02/2022'),
                  'value': np.array(constant_sales * 52) + np.random.normal(0, 2, 364),
                  'product_id': [1] * 364})
product_2 = pd.DataFrame({'date': pd.date_range(start='01/04/2021', end='01/02/2022'),
                  'value': np.array(weekly_sales * 52) + np.random.normal(0, 2, 364),
                  'product_id': [2] * 364})
product_3 = pd.DataFrame({'date': pd.date_range(start='01/04/2021', end='01/02/2022'),
                  'value': np.array(constant_sales * 52) + np.random.normal(0, 2, 364),
                  'product_id': [3] * 364})

X = pd.concat([product_0, product_1])
test = pd.concat([product_2, product_3])

features = extract_features(timeseries_container=X,
                            column_id="product_id", column_value='value', column_sort="date",
                            default_fc_parameters=MinimalFCParameters())

features = features.rename_axis('product_id').reset_index()
X = X.merge(features, on=['product_id'])
X['day_of_week'] = X['date'].apply(lambda date: date.day_of_week)

features = extract_features(timeseries_container=test,
                            column_id="product_id", column_value='value', column_sort="date",
                            default_fc_parameters=MinimalFCParameters())

features_list = features.columns.to_list() + ['day_of_week']

features = features.rename_axis('product_id').reset_index()
test = test.merge(features, on=['product_id'])
test['day_of_week'] = test['date'].apply(lambda date: date.day_of_week)

model = None
def train_and_predict(flist, test_set):
    train_data = lgb.Dataset(X[flist], label=X['value'])
    model = lgb.train({}, train_data)
    pred = model.predict(test_set[flist])
    return pred

pred = train_and_predict(features_list, test[test['product_id'] == 2])
print(pred[:7])
# -> [20.43 25.08 30.39 24.61 20.04 30.14 -0.19]
print(weekly_sales)
# -> [20, 25, 30, 25, 20, 30, 0]

pred = train_and_predict(features_list, test[test['product_id'] == 3])

print(pred[:7])
# -> [10.17  9.75 10.06  9.95  9.94  10.14  10.28]
print(constant_sales)
# -> [10, 10, 10, 10, 10, 10, 10]

# plot features importance
#plt.style.use('grayscale')
#lgb.plot_importance(model, importance_type='gain')
#plt.savefig('ts_fresh_gain.png')
#plt.show()

# without tsfresh
pred = train_and_predict(['day_of_week'], test[test['product_id'] == 2])
print(pred[:7])
# -> [14.90 17.50 20.01 17.38  15.07 20.18 5.27]
print(weekly_sales)

pred = train_and_predict(['day_of_week'], test[test['product_id'] == 3])

print(pred[:7])
# -> [14.90 17.50 20.01 17.38  15.07 20.18  5.27]
print(constant_sales)
