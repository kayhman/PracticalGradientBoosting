from sklearn.metrics import accuracy_score

y_pred = [0, 1, 1, 2]
y_true = [0, 1, 2, 1]

print(accuracy_score(y_true, y_pred))
# -> 0.5
