# tree\_ensemble\_weak.py
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

print(strong_predictor({'x': 1}))  # 1
print(strong_predictor({'x': 2}))  # 2
print(strong_predictor({'x': 3}))  # 3
print(strong_predictor({'x': 4}))  # 4
