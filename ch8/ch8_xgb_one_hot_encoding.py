# ch8_xgb_one_hot_encoding.p
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# With pandas
serie = pd.Series(['Category 1', 'Category 2', 'Category 1', 'Category 3', 'Category 1'])
print(pd.get_dummies(serie))
# ->    Category 1  Category 2  Category 3
# -> 0           1           0           0
# -> 1           0           1           0
# -> 2           1           0           0
# -> 3           0           0           1
# -> 4           1           0           0
