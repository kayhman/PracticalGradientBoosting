# tree\_ensemble.py
from decision_node import DecisionNode
from decision_tree import DecisionTree

L0 = DecisionNode('L0', lambda d: d['x'] < 2)
L1_L = DecisionNode('L1_L', None, 1)
L1_R = DecisionNode('L1_R', lambda d: d['x'] < 3,
                    None)
L2_L = DecisionNode('L2_L', None, 2)
L2_R = DecisionNode('L2_R', lambda d: d['x'] < 4,
                    None)
L3_L = DecisionNode('L3_L', None, 3)
L3_R = DecisionNode('L3_R', None, 4)

L0.add_left_node(L1_L)
L0.add_right_node(L1_R)

L1_R.add_left_node(L2_L)
L1_R.add_right_node(L2_R)

L2_R.add_left_node(L3_L)
L2_R.add_right_node(L3_R)

tree = DecisionTree(L0)
print(tree.predict({'x': 1}))  # 1
print(tree.predict({'x': 2}))  # 2
print(tree.predict({'x': 3}))  # 3
print(tree.predict({'x': 4}))  # 4
