# decision\_tree\_example.py
from decision_node import DecisionNode
from decision_tree import DecisionTree

root = DecisionNode('root',
                    lambda d: d['A'] < 20.0)
root_left = DecisionNode('root_left', None, 2)
root_right = DecisionNode(
    'root_right', lambda d: d['B'] < 666.0, None)
left_left = DecisionNode('left_left', None, 4)
left_right = DecisionNode('left_right', None, 6)

root.add_left_node(root_left)
root.add_right_node(root_right)

root_right.add_left_node(left_left)
root_right.add_right_node(left_right)

tree = DecisionTree(root)
print(tree.predict({'A': 18, 'B': 5}))  # 2
print(tree.predict({'A': 18, 'B': 155}))  # 2
print(tree.predict({'A': 23, 'B': 555}))  # 4
print(tree.predict({'A': 23, 'B': 777}))  # 6
