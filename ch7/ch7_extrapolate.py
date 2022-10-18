import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

# Create an XGBoost model
model = XGBRegressor(n_estimators=250)

# Create time serie timestamp indices
ts = np.linspace(0, 10, 100)
X = pd.DataFrame({'ts': ts})

# Generate signal to predict using a simple linear system
y = ts * 6.66

# Train XGBoost model
model.fit(X, y)

# Create prediction inputs. Start with timestamp indices
# Shift the initial time range by 0.05 to force interpolation and augment if to force extrapolation
x_preds = pd.DataFrame({'ts': list(ts + 0.05) + [11, 12, 13, 14, 15]})
preds = model.predict(x_preds)
# Plot results.
# XGBoost cannot extrapolate, and keep using the same value for prediction in the future
plt.style.use('grayscale')
plt.plot(x_preds, x_preds['ts'] * 6.66, label='Valeurs réelles')
plt.plot(x_preds, preds, label='Valeurs prédites par XGBoost')
plt.legend()

plt.savefig('extrapolate.png')
plt.show()
