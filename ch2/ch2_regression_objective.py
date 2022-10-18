# regression\_objective.py
import numpy as np

from decision_node import DecisionNode
from decision_tree import DecisionTree

T0_L0 = DecisionNode('L0', lambda d: d['x'] < 2)
T0_L1_L = DecisionNode('L1_L', None, 8)
T0_L1_R = DecisionNode('L1_R', None, 9)

T1_L0 = DecisionNode('L0', lambda d: d['x'] < 3)
T1_L1_L = DecisionNode('L1_L', None, -6)
T1_L1_R = DecisionNode('L1_R', None, -5)

T2_L0 = DecisionNode('L0', lambda d: d['x'] < 4)
T2_L1_L = DecisionNode('L1_L', None, -1)
T2_L1_R = DecisionNode('L1_R', None, 0)

T0_L0.add_left_node(T0_L1_L)
T0_L0.add_right_node(T0_L1_R)

T1_L0.add_left_node(T1_L1_L)
T1_L0.add_right_node(T1_L1_R)

T2_L0.add_left_node(T2_L1_L)
T2_L0.add_right_node(T2_L1_R)

T0 = DecisionTree(T0_L0)
T1 = DecisionTree(T1_L0)
T2 = DecisionTree(T2_L0)

strong_predictor = lambda x: T0.predict(
    x) + T1.predict(x) + T2.predict(x)

squared_error = lambda y_true, y_pred: (y_true -
                                        y_pred)**2
x_train = [1, 2, 3, 4]
y_train = [1, 2, 3, 4]

T0_mse = np.mean([
    squared_error(x, T0.predict({'x': x}))
    for x, y in zip(x_train, y_train)
])
T1_mse = np.mean([
    squared_error(x, T1.predict({'x': x}))
    for x, y in zip(x_train, y_train)
])
T2_mse = np.mean([
    squared_error(x, T2.predict({'x': x}))
    for x, y in zip(x_train, y_train)
])

strong_mse = np.mean([
    squared_error(x, strong_predictor({'x': x}))
    for x, y in zip(x_train, y_train)
])

print(T0_mse)  # 39.75
print(T1_mse)  # 64.5
print(T2_mse)  # 11.25
print(strong_mse)  # 0.0
