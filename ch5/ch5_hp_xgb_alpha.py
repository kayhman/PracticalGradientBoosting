# hp_xgb_alpha.py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
from xgboost import plot_tree

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Régularisation de type L2
model = XGBClassifier(n_estimators=20,
                      learning_rate=0.3,
                      max_depth=3,
                      gamma=0,
                      reg_alpha=0,
                      reg_lambda=2,
                      eval_metric='mlogloss',
                      use_label_encoder=False,
                      num_class=3)

model.fit(X_train, y_train)
preds = model.predict(X_test)
print(confusion_matrix(y_test, preds))
# -> [[10  0  0]
# -> [ 0  9  0]
# -> [ 0  0 11]]

# récupération des poids
dtf = model.get_booster().trees_to_dataframe()
dtf = dtf[dtf.Feature == 'Leaf']
# récupération des poids proches de zéro
dtf = dtf.sort_values(by=['Tree', 'Node', 'Gain'])
print('zero nodes:', dtf[abs(dtf.Gain) == 0.0].shape) # -> zero nodes: (0, 10)
print(model.get_params())

plot_tree(model, num_trees=5)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('model_l2.png')
fig.show()



# Régularisation de type L1 - modèle creux
model = XGBClassifier(n_estimators=20,
                      learning_rate=0.3,
                      max_depth=3,
                      gamma=0,
                      reg_alpha=2,
                      reg_lambda=0,
                      eval_metric='mlogloss',
                      use_label_encoder=False,
                      num_class=3)

model.fit(X_train, y_train)
preds = model.predict(X_test)
print(confusion_matrix(y_test, preds))
# -> [[10  0  0]
# -> [ 0  9  0]
# -> [ 0  0 11]]


plot_tree(model, num_trees=5)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 10)
fig.savefig('model_l1.png')
fig.show()

dtf1 = dtf

# récupération des poids
dtf = model.get_booster().trees_to_dataframe()
dtf = dtf[dtf.Feature == 'Leaf']
# récupération des poids proches de zéro
dtf = dtf.sort_values(by=['Tree', 'Node', 'Gain'])
plt.clf()
plt.style.use('grayscale')
plt.plot(dtf['Gain'].abs(),
         label='Valeur absolue des poids - L1')
plt.plot(dtf1['Gain'].abs(),
         linestyle='--',
         label='Valeur absolue des poids - L2')
plt.legend(loc="upper center",
           fontsize='xx-large',
           ncol=2,
           bbox_to_anchor=(0.5, 1.16))
plt.savefig('sparse_model.png')
plt.show()
print('zero nodes:', dtf[abs(dtf.Gain) == 0.0].shape)  # -> zero nodes: (65, 10)
print(model.get_params())
