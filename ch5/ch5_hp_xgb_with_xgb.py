import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

from hp_xgb_optimizer import Optimizer

def run(data, target, cfg):
    cfg_dict = {key: cfg[key] for key in cfg}
    rfr = RandomForestRegressor(**cfg_dict)

    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y) ** 2))

    # Creating root mean square error for sklearns crossvalidation
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    score = cross_val_score(rfr, data, target, cv=11, scoring=rmse_scorer, verbose=0)
    score = np.mean(score)

    return score  # Because cross_validation sign-flips the score


num_trees = UniformIntegerHyperparameter("n_estimators", 10, 50, default_value=10)
max_features = UniformIntegerHyperparameter("max_features", 1, 13, default_value=1)
min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5, default_value=0.0)
min_samples_to_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)

do_bootstrapping = CategoricalHyperparameter("bootstrap", ["true", "false"], default_value="true")
criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default_value="mse")

cs = ConfigurationSpace()
cs.add_hyperparameters([num_trees, min_weight_frac_leaf,
                        max_features, min_samples_to_split,
                        min_samples_in_leaf, max_leaf_nodes,
                        criterion, do_bootstrapping])

boston = load_boston()
max_intensification = 25

# optimizer = Optimizer(run, 50, max_intensification, lambda: CatBoostRegressor(cat_features=['criterion', 'do_bootstrapping']))
optimizer = Optimizer(lambda cfg: run(boston.data, boston.target, cfg),
                      50, 250,
                      lambda: CatBoostRegressor(cat_features=['criterion', 'bootstrap']),
                      cs)
optimizer.optimize()


# XGBoost doesn't support categorical parameters
cs = ConfigurationSpace()
cs.add_hyperparameters([num_trees, min_weight_frac_leaf,
                        max_features, min_samples_to_split,
                        min_samples_in_leaf, max_leaf_nodes])
optimizer_xgb = Optimizer(lambda cfg: run(boston.data, boston.target, cfg),
                          50, 250,
                          XGBRegressor,
                          cs)
optimizer_xgb.optimize()

optimizer_cat = Optimizer(lambda cfg: run(boston.data, boston.target, cfg),
                          50, 250,
                          CatBoostRegressor,
                          cs)
optimizer_cat.optimize()


cfg_traj = [cfg for cfg in optimizer.trajectory]
plt.style.use('grayscale')
plt.plot([-optimizer.scores[cfg] if cfg in cfg_traj else None for cfg in optimizer.cfgs], 'o')
plt.plot([-optimizer.scores[cfg] for cfg in optimizer.cfgs], '-')

cfg_traj = [cfg for cfg in optimizer_xgb.trajectory]
plt.plot([-optimizer_xgb.scores[cfg] if cfg in cfg_traj else None for cfg in optimizer_xgb.cfgs], 'o')
plt.plot([-optimizer_xgb.scores[cfg] for cfg in optimizer_xgb.cfgs], '-.')

cfg_traj = [cfg for cfg in optimizer_cat.trajectory]
plt.plot([-optimizer_cat.scores[cfg] if cfg in cfg_traj else None for cfg in optimizer_cat.cfgs], 'o')
plt.plot([-optimizer_cat.scores[cfg] for cfg in optimizer_cat.cfgs], ':')

print('Catboost - cat', optimizer.best_cfg, optimizer.best_score)
print('XGBoost', optimizer_xgb.best_cfg, optimizer_xgb.best_score)
print('Catboost', optimizer_cat.best_cfg, optimizer_cat.best_score)

plt.title('scores')
plt.legend(['Configurations testées(CatBoost - cat)', 'Configurations retenues (CatBoost - cat)',
            'Configuration testées (XGBoost)', 'Configurations retenues (XGBoost)',
            'Configurations testées (CatBoost)', 'Configurations retenues (CatBoost)'])
plt.xlabel('Nombre d \'iterations')
plt.ylabel('Score')
plt.savefig('xgb_with_xgb.png')
plt.show()
