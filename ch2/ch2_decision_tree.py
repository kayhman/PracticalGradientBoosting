# decision\_tree.py
class DecisionTree:
    """
    A DecisionTree is a model that
    provides predictions depending on input.
    A Prediction is the sum of the leaves' values,
    for those leaves that were activated
    by the input
    """

    def __init__(self, root):
        """
        A DecisionTree is defined by an objective,
        a number of estimators and a max depth.
        """
        self.root = root

    def predict(self, data):
        child = self.root
        while child and not child.is_leaf():
            child = child.next(data)
        return child.value
