# ch4_shap_global.py
import xgboost
import shap
from matplotlib import pyplot as plt

X, y = shap.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values)
plt.show()
