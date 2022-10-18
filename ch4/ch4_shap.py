import xgboost
import shap
from matplotlib import pyplot as plt

# train an XGBoost model
X, y = shap.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
#shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
plt.show()
