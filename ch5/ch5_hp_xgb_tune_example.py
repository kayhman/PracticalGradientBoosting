import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ray import tune


def train_iris(config):
    data, labels = sklearn.datasets.load_iris(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)

    results = {}
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        evals_result=results,
        verbose_eval=False)

    accuracy = 1. - results["eval"]["error"][-1]
    tune.report(mean_accuracy=accuracy, done=True)


config = {
    "eval_metric": ["logloss", "error"],
    "max_depth": tune.randint(1, 9),
    "min_child_weight": tune.choice([1, 2, 3]),
    "subsample": tune.uniform(0.5, 1.0),
    "eta": tune.loguniform(1e-4, 1e-1)
}
analysis = tune.run(
    train_iris,
    resources_per_trial={"cpu": 8},
    config=config,
    num_samples=10)

print(analysis.get_best_config(metric="mean_accuracy", mode="min"))
# {'eval_metric': ['logloss', 'error'], 'max_depth': 3, 'min_child_weight': 2, 'subsample': 0.7810857640547034, 'eta': 0.012662938165536894}
