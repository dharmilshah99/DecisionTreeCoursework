import numpy as np
import node

def predict(decision_tree, x):
    """Performs prediction on some samples using a decision tree.

    Args:
        decision_tree (Node): Decision tree.
        x (np.ndarray): Instances, numpy array with shape (N,K).

    Returns:
        y (np.ndarray): Predicted class labels, numpy array with shape (N,).
    """
    y = np.zeros(len(x))
    for idx, inst in enumerate(x):
        # Traverse tree until leaf
        curr_node = decision_tree
        while not curr_node.is_root():
            if inst[curr_node.attribute] < curr_node.value:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        y[idx] = curr_node.label
    return y



def compute_accuracy(y_gold, y_prediction):
    """Compute the accuracy given the ground truth and predictions.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels.
        y_prediction (np.ndarray): the predicted labels.

    Returns:
        float : the accuracy.
    """

    assert len(y_gold) == len(y_prediction)

    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0

if __name__ == "__main__":
    pass
