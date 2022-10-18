# ch4_feature_importance.py
import matplotlib.pyplot as plt
import pandas as pd
from jax import grad, jacfwd, jacrev, jit
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random, pickle

class DecisionNode:
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
        return (not self.left) and (not self.right)

    def next(self, data):
        cond = self.condition(data)
        if cond:
            return self.left
        else:
            return self.right

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
    def __init__(self, nb_estimators, max_depth,
                 gamma=0, lbda=0):
        self.roots = [DecisionNode(f'root_{esti}', None, 0.0) for esti in range(0, nb_estimators)]
        self.lbda = lbda
        self.gamma = gamma
        self.objective = lambda y_pred, y_true: np.dot(y_true - y_pred, (y_true - y_pred).T)
        self.grad = lambda y_pred, y_true: -2 * (y_true - y_pred)
        self.hessian = lambda y_pred, y_true: np.array([2] * y_pred.shape[0])
        self.max_depth = max_depth
        self.base_score = None
        self.feature_importances = {'weights': {},
                                    'gains': {},
                                    'cover': {}}


    def _create_condition(self, col_name, split_value):
        return lambda dta : dta[col_name] < split_value

    def _add_child_nodes(self, node, nodes,
                         node_x, node_y,
                         split_value, split_column,
                         nb_nodes,
                         left_w, right_w, prev_w):
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
        self.base_score = y_train.mean()
        self.roots[0].value = self.base_score
        node_count = 0
        for tree_idx, tree_root in enumerate(self.roots):
            # store current node (currenly a lead), x_train and node leaf weight
            nodes = [(tree_root, x_train.copy(), y_train.copy(), 0.0)]
            nb_nodes = 0
            real_depth = 0
            # Add node to tree using bfs
            while nodes:
                node, node_x, node_y, prev_w = nodes.pop(0)
                node_x['pred'] = self.predict(node_x)
                best_gain = None
                cols = x_train.columns.tolist()
                for split_column in cols:
                    split, split_value, left_w, \
                        right_w, left_gain, right_gain, \
                        gain = self._find_best_split(split_column,
                                                          node_x, node_y,
                                                          nb_nodes)
                    if best_gain is None:
                        best_gain = gain
                    if gain >= best_gain:
                        best_split_column = split_column
                        best_split = split
                        best_split_value = split_value
                        best_left_w = left_w
                        best_right_w = right_w
                        best_prev_w = prev_w
                        best_left_gain = left_gain
                        best_right_gain = right_gain
                if best_split != -1 and node.depth < self.max_depth:
                    self._add_child_nodes(node, nodes,
                                          node_x, node_y,
                                          best_split_value, best_split_column,
                                          node_count,
                                          best_left_w, best_right_w, best_prev_w)
                    self._update_feature_importances(best_split_column,
                                                     best_left_gain + best_right_gain,
                                                     node_x.shape[0])
                    node_count += 2
                    real_depth += 1
                nb_nodes += 1
                if nb_nodes >= 2**self.max_depth-1:
                    break


    def _update_feature_importances(self, feature_name, gain, cover):
        self.feature_importances['gains'][feature_name] = self.feature_importances['gains'].get(feature_name, []) + [float(gain)]
        self.feature_importances['cover'][feature_name] = self.feature_importances['cover'].get(feature_name, []) + [float(cover)]
        self.feature_importances['weights'][feature_name] = self.feature_importances['weights'].get(feature_name, 0) + 1

    def _gain_and_weight(self, x_train, y_train, nb_nodes):
        pred = x_train['pred'].values
        G_i = self.grad(pred, y_train.values).sum()
        H_i = self.hessian(pred, y_train.values).sum()
        return -0.5 * G_i * G_i / (H_i + self.lbda) + self.gamma * nb_nodes, -G_i / (H_i + self.lbda)

    def _find_best_split(self, col_name, node_x, node_y, nb_nodes):
        x_sorted = node_x.sort_values(by=col_name)
        y_sorted = node_y[x_sorted.index]
        current_gain, _ = self._gain_and_weight(x_sorted, node_y, nb_nodes)
        gain = 0.0
        best_split = -1
        split_value, best_left_w, best_right_w, best_left_gain, best_right_gain = None, None, None, None, None
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
                gain = current_gain - (left_gain + right_gain)
                best_split = split_idx
                split_value = x_sorted[col_name].iloc[split_idx]
                best_left_w = left_w
                best_right_w = right_w
                best_left_gain = left_gain
                best_right_gain = right_gain
        return best_split, split_value, best_left_w, best_right_w, best_left_gain, best_right_gain, gain

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


x_train = pd.DataFrame({"A" : [3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]})
y_train = pd.DataFrame({"Y" : [3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]})

tree = DecisionEnsemble(1, 3)
tree.fit(x_train, y_train['Y'])
tree.dump('identity.rst')
pred = tree.predict(pd.DataFrame({'A': [1., 2., 3., 4., 5., 6., 7.]}))
print(pred) #-> [1. 2. 3. 4. 5. 6. 7.]


boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

y_train = pd.DataFrame({"Y" : y_train})
x_train = pd.DataFrame(x_train, columns=boston.feature_names)
x_test = pd.DataFrame(x_test, columns=boston.feature_names)
tree = DecisionEnsemble(40, 3)
print(boston.feature_names)

tree.fit(x_train, y_train['Y'])
pred = tree.predict(x_test)
print(pred)
#-> [21.7, 28.47, 21.7 28.47, 13.06, 28.47, ...


mean = {k: abs(np.mean(v)) for k,v in tree.feature_importances['gains'].items()}
mean = {k: v for k, v in sorted(mean.items(), key=lambda item: item[1])}
print(mean)
plt.barh(list(mean.keys()), mean.values())
plt.savefig('feature_importances.png')
plt.show()
# -> {'LSTAT': -6905.327611957848, 'CRIM': -155.85076530601873, 'PTRATIO': -39.75703359393743, 'TAX': -117.44408069160463, 'DIS': -162.58054656898864, 'RM': -466.7232793454371, 'B': -104.72029666312362, 'RAD': -117.37806885091399, 'AGE': -41.4475396048749, 'NOX': -58.25049793614738, 'INDUS': -83.88902121942156, 'ZN': -43.02020051516297, 'CHAS': -11.362683360304638}
with open('feature_importances.pkl', 'wb') as file:
    pickle.dump(tree.feature_importances, file)
