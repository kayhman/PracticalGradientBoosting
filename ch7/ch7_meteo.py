import numpy as np
import pandas as pd
from xgboost import XGBRegressor

temperature = pd.DataFrame({'date': pd.date_range(start='01/04/2021', end='01/02/2022'),
                            'temperature': np.random.normal(14, 5, 364)})

temperature['temperature_J-1'] = temperature['temperature'].shift(1)

model = XGBRegressor(n_estimators=1250, max_depth=5)

model.fit(temperature[['temperature_J-1']], temperature['temperature'])

pred = model.predict(pd.DataFrame({'temperature_J-1' : [17, 18, 19]}))

print(pred)
# -> [20.2686   15.555077 14.99681 ]
