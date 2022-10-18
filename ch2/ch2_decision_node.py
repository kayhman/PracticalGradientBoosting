# decision node.py
class DecisionNode:
    """
    Node decision class.
    This is a simple binary node,
    with potentially two children:
      left and right
    Left node is returned when condition is true
    False node is returned when condition is false
    """

    def __init__(self,
                 name,
                 condition,
                 value=None):
        self.name = name
        self.condition = condition
        self.value = value
        self.left = None
        self.right = None

    def add_left_node(self, left):
        self.left = left

    def add_right_node(self, right):
        self.right = right

    def is_leaf(self):
        """
        Node is a leaf if it has no child
        """
        return (not self.left) and (
            not self.right)

    def next(self, data):
        """
        Return next node depending on
        data and node condition
        """
        cond = self.condition(data)
        if cond:
            return self.left
        else:
            return self.right
