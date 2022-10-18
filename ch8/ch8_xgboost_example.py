# ch8_xgboost_example.py
from xgboost import XGBClassifier
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

categorical_features_indices = np.where(train.dtypes != 'int64')[0]
categorical_features = list(train.columns[np.where(train.dtypes != np.int64)[0]])

one_hot_cols = []
for col in categorical_features:
    s = train[col].unique()

    one_hot_df = pd.get_dummies(s, prefix='%s_' % col)
    one_hot_df[col] = s

    train = train.merge(one_hot_df, on=[col], how="left")

    one_hot_cols += list(set(one_hot_df.columns.tolist()) - set([col]))

    s = test[col].unique()

    one_hot_df = pd.get_dummies(s, prefix='%s_' % col)
    one_hot_df[col] = s

    test = test.merge(one_hot_df, on=[col], how="left")

X = train[one_hot_cols]
y = train[to_predict]

test['native-country__ Holand-Netherlands'] = 0

X_test = test[one_hot_cols]
y_test = test[to_predict]

X.rename(columns={'income__ >50K': 'income__ GT 50K',
                  'income__ <=50K': 'income__ LT 50K'},
         inplace=True)
X_train, X_validation, y_train, y_validation = train_test_split(X, y,
                                                                train_size=0.7, random_state=42)


model = XGBClassifier()
model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])


pred = model.predict(X_test)
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
