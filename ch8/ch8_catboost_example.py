# ch8_catboost_example.py
from catboost import CatBoostClassifier
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

categorical_features_indices = np.where(train.dtypes != np.float)[0]
model = CatBoostClassifier(cat_features=categorical_features_indices)
model.fit(X_train, y_train, eval_set=(X_validation, y_validation))

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
