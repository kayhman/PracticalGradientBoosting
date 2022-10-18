# train_ensemble_tree.py
from collections import OrderedDict as OD
import matplotlib.pyplot as plt
import pandas as pd
from jax import grad, jacfwd, jacrev, jit
import jax.numpy as jnp
import numpy as np

import random

class DecisionNode:
    """
    Node decision class.
    This is a simple binary node, with potentially two childs: left and right
    Left node is returned when condition is true
    False node is returned when condition is false<
    """
    def __init__(self, name, condition,
                 value=None, depth=0,
                 label=None):
        self.name = name
        self.condition = condition
        self.label = label
        self.value = value
        self.left = None
        self.right = None
        self.depth = depth

    def add_left_node(self, left):
        self.left = left

    def add_right_node(self, right):
        self.right = right

    def is_leaf(self):
        """
        Node is a leaf if it has no child
        """
        return (not self.left) and (not self.right)

    def next(self, data):
        """
        Return next code depending on data and node condition
        """
        cond = self.condition(data)
        if cond:
            return self.left
        else:
            return self.right

def hessian(fun):
  return jit(jacfwd(jacrev(fun)))

def tree_to_rst(roots, filename):
    with open(filename, 'w') as rst:
        rst.write('graph TD' + '\n')
        rst.write('\t' + 'classDef node fill:#fff,stroke:#333,stroke-width:1px;' + '\n')
        for root in roots:
            root.label = f'{root.label}<br>{root.value}'
            nodes = [root]
            edges = []
            while nodes:
                node = nodes.pop()
                if node.is_leaf():
                    rst.write(f'\t{node.name}("{node.value}")\n')
                else:
                    rst.write(f'\t{node.name}("{node.label or node.value}")\n')
                if node.left:
                    nodes.append(node.left)
                    edges.append((node.name, node.left.name))
                if node.right:
                    nodes.append(node.right)
                    edges.append((node.name, node.right.name))
            for begin, end in edges:
                rst.write(f'\t{begin} --> {end}\n')

class DecisionEnsemble:
    """
    A DecisionEnsemble is a model that provides predictions depending on input.
    Prediction is the sum of the values attached to leaf activated by input
    """
    def __init__(self, objective, nb_estimators, max_depth,
                 gamma=0, lbda=0):
        """
        A DecisionEnsemble is defined by an objective, a number of estimators and a max depth.
        """
        self.roots = [DecisionNode(f'root_{esti}', None, 0.0) for esti in range(0, nb_estimators)]
        self.objective = objective
        self.lbda = lbda
        self.gamma = gamma
        self.grad = grad(self.objective)
        self.hessian = hessian(self.objective)
        self.max_depth = max_depth
        self.base_score = None


    def _create_condition(self, col_name, split_value):
        """
        Create a closure that capture split value
        """
        return lambda dta : dta[col_name] < split_value

    def _pick_columns(self, columns, x_dtf):
        cols = []
        for col in columns:
            if len(x_dtf[col].unique()) > 1:
                cols.append(col)
        if cols:
            return random.choice(cols)
        else:
            return None

    def _add_child_nodes(self, node, nodes,
                         node_x, node_y,
                         split_value, split_column,
                         nb_nodes,
                         left_w, right_w, prev_w):
        #node.name = f'N{node.depth}({split_column} < {split_value})'
        node.condition = self._create_condition(split_column, split_value) # we must create a closure to capture split_value copy
        node.label = f'{split_column} < {split_value}'
        node.add_left_node(DecisionNode(f'left_{nb_nodes}',
                                        None, left_w + prev_w,
                                        depth = node.depth+1))
        node.add_right_node(DecisionNode(f'right_{nb_nodes}',
                                         None, right_w + prev_w,
                                         depth = node.depth+1))
        mask = node_x[split_column] < split_value
        # Reverse order to ensure bfs
        nodes.append((node.left,
                      node_x[mask].copy(),
                      node_y[mask].copy(),
                      left_w + prev_w))
        nodes.append((node.right,
                      node_x[~mask].copy(),
                      node_y[~mask].copy(),
                      right_w + prev_w))


    def fit(self, x_train, y_train):
        """
        Fit decision trees using x_train and objective
        """
        self.base_score = y_train.mean()
        self.roots[0].value = self.base_score
        node_count = 0
        for tree_idx, tree_root in enumerate(self.roots):
            print(f'---------- root {tree_root.name} ---------------')
            # store current node (currenly a lead), x_train and node leaf weight
            nodes = [(tree_root, x_train.copy(), y_train.copy(), 0.0)]
            nb_nodes = 0
            real_depth = 0
            # Add node to tree using bfs
            while nodes:
                node, node_x, node_y, prev_w = nodes.pop(0)
                node_x['pred'] = self.predict(node_x)
                split_column = self._pick_columns(x_train.columns, node_x) # XGBoost use a smarter heuristic here
                if split_column:
                    print('look for splitting')
                    cols = x_train.columns.tolist()
                    while cols:
                        print('into', cols)
                        split_column = self._pick_columns(cols, node_x) # XGBoost use a smarter heuristic here
                        if not split_column:
                            break
                        best_split, split_value, left_w, right_w = self._find_best_split(split_column,
                                                                                         node_x, node_y,
                                                                                         nb_nodes)
                        if best_split != -1:
                            break
                        else:
                            cols.remove(split_column)
                    print('best_split', best_split)
                    if best_split != -1 and node.depth < self.max_depth:
                        self._add_child_nodes(node, nodes,
                                              node_x, node_y,
                                              split_value, split_column,
                                              node_count,
                                              left_w, right_w, prev_w)
                    node_count += 2
                    real_depth += 1
                nb_nodes += 1
                if nb_nodes >= 2**self.max_depth-1:
                    break
            print('----> real depth', real_depth, 'for tree', tree_idx, nb_nodes, 2**self.max_depth)


    def _gain_and_weight(self, x_train, y_train, nb_nodes):
        """
        Compute gain and leaf weight using automatic differentiation
        """
        pred = x_train['pred'].values
        G_i = self.grad(pred, y_train.values).sum()
        H_i = self.hessian(pred, y_train.values).sum()
        return -0.5 * G_i * G_i / (H_i + self.lbda) + self.gamma * nb_nodes, -G_i / (H_i + self.lbda)

    def _find_best_split(self, col_name, node_x, node_y, nb_nodes):
        """
        Compute best split
        """
        x_sorted = node_x.sort_values(by=col_name)
        y_sorted = node_y[x_sorted.index]
        current_gain, _ = self._gain_and_weight(x_sorted, node_y, nb_nodes)
        gain = 0.0
        best_split = -1
        split_value, best_left_w, best_right_w = None, None, None
        for split_idx in range(1, x_sorted.shape[0]):
            # skip equal value
            if split_idx <x_sorted.shape[0]-1:
                if x_sorted.iloc[split_idx][col_name] == x_sorted.iloc[split_idx+1][col_name]:
                    continue
            left_data = x_sorted.iloc[:split_idx]
            right_data = x_sorted.iloc[split_idx:]
            left_y = y_sorted.iloc[:split_idx]
            right_y = y_sorted.iloc[split_idx:]
            left_gain, left_w = self._gain_and_weight(left_data, left_y, nb_nodes)
            right_gain, right_w = self._gain_and_weight(right_data, right_y, nb_nodes)
            if current_gain - (left_gain + right_gain) > gain:
                print('prev gain', gain, col_name, split_value)
                gain = current_gain - (left_gain + right_gain)
                print('current gain', gain)
                best_split = split_idx
                split_value = x_sorted[col_name].iloc[split_idx]
                best_left_w = left_w
                best_right_w = right_w
        return best_split, split_value, best_left_w, best_right_w

    def predict(self, data):
        preds = []
        for _, row in data.iterrows():
            pred = 0.0
            for tree_idx, root in enumerate(self.roots):
                child = root
                while child and not child.is_leaf():
                    child = child.next(row)
                pred += child.value
            preds.append(pred)
        return np.array(preds) + self.base_score

    def dump(self, filename):
        tree_to_rst(self.roots, filename)


def squared_error(y_pred, y_true):
    diff = y_true - y_pred
    return jnp.dot(diff, diff.T)

x_train = pd.DataFrame({"A" : [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]})
y_train = pd.DataFrame({"Y" : [-1, -1, -1, 1.0, 1.0, 1.0, 1.0]})

tree = DecisionEnsemble(squared_error, 1, 2)
tree.fit(x_train, y_train['Y'])
tree.dump('is_positive.rst')
pred = tree.predict(pd.DataFrame({'A': [1., 2., 3., 4., 5., 6., 7.]}))
print(pred) #-> [-1. -1. -1. 1. 1. 1. 1.]
