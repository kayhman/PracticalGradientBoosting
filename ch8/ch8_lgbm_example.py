# ch8_lgbm_example.py
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

train = pd.read_csv('data/adult.csv')
test = pd.read_csv('data/adult_test.csv')

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
	    'marital-status', 'occupation', 'relationship', 'sex', 'capital-gain',
            'capital-loss', 'hour-per_week', 'native-country', 'income']
to_predict = 'income'

X = train[features]
y = train[to_predict]

X_train, X_validation, y_train, y_validation = train_test_split(X, y,
                                                                train_size=0.7, random_state=42)

categorical_features_indices = np.where(train.dtypes != 'int64')[0]
categorical_features = list(train.columns[np.where(train.dtypes != np.int64)[0]])
for feature in categorical_features:
    X_train[feature] = pd.Series(X_train[feature], dtype="category")
    X_validation[feature] = pd.Series(X_validation[feature], dtype="category")
    test[feature] = pd.Series(test[feature], dtype="category")

model = LGBMClassifier()
model.fit(X_train, y_train, eval_set=(X_validation, y_validation),
          categorical_feature=categorical_features)

pred = model.predict(test[features])
print(classification_report(test[to_predict], pred))
# Name: income, Length: 16281, dtype: object
#               precision    recall  f1-score   support
#
#        <=50K       1.00      1.00      1.00     12435
#         >50K       1.00      1.00      1.00      3846
#
#     accuracy                           1.00     16281
#    macro avg       1.00      1.00      1.00     16281
# weighted avg       1.00      1.00      1.00     16281
print(confusion_matrix(test[to_predict], pred))
# [[12435     0]
#  [    0  3846]]
print(accuracy_score(test[to_predict], pred))
# 1.0
