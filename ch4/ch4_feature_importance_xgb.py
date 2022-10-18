# ch4_feature_importance_xgb.py
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

xgb = XGBRegressor(n_estimators=100, importance_type="cover")
xgb.fit(x_train, y_train)
sorted_idx = xgb.feature_importances_.argsort()
print({boston.feature_names[idx]:xgb.feature_importances_[idx] for idx in sorted_idx})
# -> {'ZN': 0.027104674, 'INDUS': 0.036392692, 'CRIM': 0.03985535, 'RAD': 0.060898088, 'CHAS': 0.06657668, 'NOX': 0.0733411, 'RM': 0.08605623, 'PTRATIO': 0.08794409, 'B': 0.09772734, 'DIS': 0.0979996, 'AGE': 0.101690546, 'LSTAT': 0.11076217, 'TAX': 0.11365144}

plt.barh(boston.feature_names[sorted_idx], xgb.feature_importances_[sorted_idx])
plt.title("Model cover")
plt.savefig('feature_importances_cover.png')
plt.show()


xgb = XGBRegressor(n_estimators=50, importance_type="gain")
xgb.fit(x_train, y_train)
sorted_idx = xgb.feature_importances_.argsort()
print({boston.feature_names[idx]:xgb.feature_importances_[idx] for idx in sorted_idx})
# -> {'ZN': 0.0013826311, 'CHAS': 0.0027196791, 'INDUS': 0.00687292, 'RAD': 0.00882377, 'B': 0.009297607, 'AGE': 0.015537402, 'CRIM': 0.022315497, 'PTRATIO': 0.03112754, 'TAX': 0.034043133, 'DIS': 0.03962602, 'NOX': 0.050526466, 'RM': 0.35646823, 'LSTAT': 0.42125916}

plt.barh(boston.feature_names[sorted_idx], xgb.feature_importances_[sorted_idx])
plt.title("Model gain")
plt.savefig('feature_importances_gain.png')
plt.show()


xgb = XGBRegressor(n_estimators=50, importance_type="weight")
xgb.fit(x_train, y_train)
sorted_idx = xgb.feature_importances_.argsort()
print({boston.feature_names[idx]:xgb.feature_importances_[idx] for idx in sorted_idx})
# -> {'CHAS': 0.008130081, 'RAD': 0.011517615, 'ZN': 0.021680217, 'INDUS': 0.034552846, 'PTRATIO': 0.034552846, 'TAX': 0.03726287, 'NOX': 0.061653115, 'B': 0.09485095, 'LSTAT': 0.0995935, 'AGE': 0.10704607, 'DIS': 0.10704607, 'RM': 0.16192412, 'CRIM': 0.2201897}

plt.barh(boston.feature_names[sorted_idx], xgb.feature_importances_[sorted_idx])
plt.title("Model weight")
plt.savefig('feature_importances_weight.png')
plt.show()
