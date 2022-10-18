from sklearn import svm, datasets
from skopt import BayesSearchCV

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = BayesSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
print(clf.best_params_)
# OrderedDict([('C', 8), ('kernel', 'rbf')])
