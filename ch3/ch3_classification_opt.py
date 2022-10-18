# ch3_classification_opt.py
from sklearn import datasets
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

digits = datasets.load_digits()
images = digits.images
targets = digits.target

images=images.reshape(1797,8*8)

X_train, X_test, y_train, y_test = train_test_split(images, targets,
                                                    test_size=0.3, random_state=42)
eval_set = [(X_train, y_train), (X_test, y_test)]

model = LGBMClassifier(objective='multiclass',
                       n_estimators=500,
                       max_depth=10)
model.fit(X_train, y_train,
          early_stopping_rounds=10,
          eval_metric=['logloss'],
          eval_set=eval_set
)


print(model.best_iteration_)
# -> 64
y_pred=model.predict(X_test,
                     start_iteratio=0,
                     num_iteration=model.best_iteration_)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)


results = model.evals_result_
plt.style.use('grayscale')
fig, ax = plt.subplots()

epochs = len(results['training']['multi_logloss'])
x_axis = range(0, epochs)

ax.plot(x_axis, results['training']['multi_logloss'], label='Train')
ax.plot(x_axis, results['valid_1']['multi_logloss'], ':', label='Test')
ax.legend()
plt.ylabel('Multiclasses log Loss')
plt.title('Early stopping')
plt.savefig('classification.png')
plt.show()
