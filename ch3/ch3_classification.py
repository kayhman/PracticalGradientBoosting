# ch3_classification.py
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
                                                    test_size=0.3,
                                                    random_state=42)

model = LGBMClassifier(objective='multiclass')
model.fit(X_train, y_train)

print(model.best_iteration_)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)
