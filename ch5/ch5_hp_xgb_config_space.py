from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

# add some dimensions to explore for XGBoost
num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default_value=10)
max_features = UniformIntegerHyperparameter("max_features", 1, 100, default_value=1)
min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default_value=0.0)
min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default_value=2)
min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default_value=1)
max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)

# add some categorical dimensions
do_bootstrapping = CategoricalHyperparameter("do_bootstrapping", ["true", "false"], default_value="true")
criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default_value="mse")

# create a configururation spaceé
cs = ConfigurationSpace()
cs.add_hyperparameters([num_trees, min_weight_frac_leaf,
                        max_features, min_samples_to_split,
                        min_samples_in_leaf, max_leaf_nodes, criterion, do_bootstrapping])

# sample configuration space
cs.sample_configuration()
# > Configuration:
#    criterion, Value: 'mae'
#    do_bootstrapping, Value: 'false'
#    max_features, Value: 36
#    max_leaf_nodes, Value: 170
#    min_samples_in_leaf, Value: 19
#    min_samples_to_split, Value: 9
#    min_weight_frac_leaf, Value: 0.018895139352121226
#    num_trees, Value: 37
