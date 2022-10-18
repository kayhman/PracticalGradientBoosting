import pandas as pd
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

n_features = 40
X1, y1 = make_regression(n_samples=1000,
                         n_features=n_features,
                         n_informative=10,
                         noise=1,
                         random_state=42)

X2, y2 = make_regression(n_samples=1000,
                         n_features=n_features,
                         n_informative=n_features,
                         noise=1,
                         random_state=42)

X1 = pd.DataFrame(X1, columns=[str(i) for i in range(0, n_features)])
X2 = pd.DataFrame(X2, columns=[str(i) for i in range(0, n_features)])

y1 = pd.DataFrame(y1, columns=['y'])
y2 = pd.DataFrame(y2, columns=['y'])

X1['VendorID'] = 1
X2['VendorID'] = 2

X = pd.concat([X1, X2])
y = pd.concat([y1, y2])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
eval_set = [(X_test, y_test)]

v1_index = list(X_test[X_test.VendorID == 1].index)
v2_index = list(X_test[X_test.VendorID == 2].index)

v1_index_train = list(X_train[X_train.VendorID == 1].index)
v2_index_train = list(X_train[X_train.VendorID == 2].index)

def multi_resolution_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    metrics = []
    mae_vendorID1 = mean_absolute_error(y_true[v1_index], y_pred[v1_index])
    mae_vendorID2 = mean_absolute_error(y_true[v2_index], y_pred[v2_index])
    return [('mae_vendorID1', mae_vendorID1), ('mae_vendorID2', mae_vendorID2)]

model= XGBRegressor(n_estimators=500,
                    max_depth=2)
model.fit(X_train, y_train,
          eval_metric=multi_resolution_metric,
          eval_set=eval_set)
results = model.evals_result()

epochs = len(results['validation_0']['mae_vendorID1'])
x_axis = range(0, epochs)

plt.style.use('grayscale')
plt.plot(x_axis, results['validation_0']['mae_vendorID1'], label='VendorID == 1')
plt.plot(x_axis, results['validation_0']['mae_vendorID2'], label='VendorID == 2')
plt.legend()
plt.ylabel('mae')
plt.title('Multi-modèles MAE')
plt.savefig('multi_resolution_metric.png')
#plt.show()


pred1 = model.predict(X_test[X_test.VendorID == 1])
pred2 = model.predict(X_test[X_test.VendorID == 2])
pred1_opt = model.predict(X_test[X_test.VendorID == 1],  iteration_range=(0, 50))
print(mean_absolute_error(y_test.iloc[v1_index], pred1))
# -> 89.65769974171772
print(mean_absolute_error(y_test.iloc[v2_index], pred2))
#-> 156.0964928287915
print(mean_absolute_error(y_test.iloc[v1_index], pred1_opt))
# -> 85.22945890509138
