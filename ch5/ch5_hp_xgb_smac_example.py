import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn import datasets

from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario


iris = datasets.load_iris()

def xgboost_from_cfg(cfg):
    clf = xgb.XGBClassifier(**cfg, random_state=0)
    scores = cross_val_score(clf, iris.data, iris.target)
    return 1 - np.mean(scores)

cs = ConfigurationSpace()

max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=3)
cs.add_hyperparameter(max_depth)

learning_rate = UniformFloatHyperparameter("learning_rate", 0.01, 1.0, default_value=1.0, log=True)
cs.add_hyperparameter(learning_rate)

max_features = UniformIntegerHyperparameter("gamma", 0, 10, default_value=4)
cs.add_hyperparameters([max_features])

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": 10,
                     "cs": cs,
                     "deterministic": "true",
                     "wallclock_limit": 120})

smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(0), tae_runner=xgboost_from_cfg)

incumbent = smac.optimize()

print(incumbent)
#Configuration:
#  gamma, Value: 1
#  learning_rate, Value: 0.014103913355442192
#  max_depth, Value: 7
