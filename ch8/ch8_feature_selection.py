# ch8_feature_selection.py
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import time

nb_samples = [100, 1000, 2000, 4000, 6000, 8000,
              10000, 20000, 40000, 60000, 80000, 100000]
nb_features = [10, 20, 30, 40, 50, 75, 80, 100, 200, 500]

def train(method):
    elapsed = []
    for n_samples in nb_samples:
        x_train, y_train = make_regression(n_samples=n_samples,
                                           n_features=20,
                                           noise=1,
                                           random_state=42)
        model = XGBRegressor(tree_method=method)
        t = time.process_time()
        model.fit(x_train, y_train)
        elapsed_time = time.process_time() - t
        elapsed.append(elapsed_time)
        print('Modèle entraîné en ', elapsed_time, 's avec', method)
    return elapsed


def train_features(method):
    elapsed = []
    for n_features in nb_features:
        x_train, y_train = make_regression(n_samples=10000,
                                           n_features=n_features,
                                           noise=1,
                                           random_state=42)
        model = XGBRegressor(tree_method=method)
        t = time.process_time()
        model.fit(x_train, y_train)
        elapsed_time = time.process_time() - t
        elapsed.append(elapsed_time)
        print('Modèle entraîné en ', elapsed_time, 's avec', method)
    return elapsed

exact_time = train('exact')
approx_time = train('approx')
hist_time = train('hist')

plt.style.use('grayscale')
plt.plot(nb_samples, exact_time, label='Méthode exacte')
plt.plot(nb_samples, approx_time, label='Méthode approx')
plt.plot(nb_samples, hist_time, label='Méthode hist')

plt.xlabel('Nombre de lignes')
plt.ylabel('Temps de calcul')

plt.legend(loc='upper left')
plt.savefig('xgb_fs.png')
plt.show()

exact_time = train_features('exact')
approx_time = train_features('approx')
hist_time = train_features('hist')

plt.style.use('grayscale')
plt.plot(nb_features, exact_time, label='Méthode exacte')
plt.plot(nb_features, approx_time, label='Méthode approx')
plt.plot(nb_features, hist_time, label='Méthode hist')

plt.xlabel('Nombre de features')
plt.ylabel('Temps de calcul')

plt.legend(loc='upper left')
plt.savefig('xgb_fs_feature.png')
plt.show()
