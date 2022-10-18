# ch3_overfit.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


from xgboost import XGBClassifier
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100,
                           n_informative=5,
                           n_classes=3,
                           random_state=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

eval_set = [(X_train, y_train), (X_test, y_test)]
model = XGBClassifier(use_label_encoder=False,
                      verbose=True,
                      num_class=3)

model.fit(X_train, y_train,
          eval_metric=['mlogloss'],
          eval_set=eval_set)
# [0]     validation_0-mlogloss:0.80954   validation_1-mlogloss:0.91414
# [1]     validation_0-mlogloss:0.61295   validation_1-mlogloss:0.80010
# [2]     validation_0-mlogloss:0.46815   validation_1-mlogloss:0.72442
# [3]     validation_0-mlogloss:0.36609   validation_1-mlogloss:0.66446
# [4]     validation_0-mlogloss:0.29383   validation_1-mlogloss:0.64616
# [5]     validation_0-mlogloss:0.23800   validation_1-mlogloss:0.61983
# [6]     validation_0-mlogloss:0.19832   validation_1-mlogloss:0.59875
# [7]     validation_0-mlogloss:0.16526   validation_1-mlogloss:0.58952
# [8]     validation_0-mlogloss:0.14131   validation_1-mlogloss:0.57995
# [9]     validation_0-mlogloss:0.12145   validation_1-mlogloss:0.57413
# [10]    validation_0-mlogloss:0.10602   validation_1-mlogloss:0.58198

results = model.evals_result()

plt.style.use('grayscale')
fig, ax = plt.subplots()

epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], ':', label='Test')
ax.legend()
plt.ylabel('Multiclasses log Loss')
plt.title('Sur-apprentissage')
plt.savefig('overfit.png')
plt.show()
