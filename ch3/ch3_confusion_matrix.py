from sklearn.metrics import confusion_matrix

y_pred = [0, 1, 1, 2]
y_true = [0, 1, 2, 1]

print(confusion_matrix(y_true, y_pred))
# -> [[1 0 0]
#     [0 1 1]
#     [0 1 0]]
