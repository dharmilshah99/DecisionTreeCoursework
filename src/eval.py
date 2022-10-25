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
        y_gold (np.ndarray): Ground truth/Gold standard labels.
        y_prediction (np.ndarray): Predicted labels.

    Returns:
        float: the accuracy.
    """

    assert len(y_gold) == len(y_prediction)

    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0

def generate_confusion_matrix(y_gold, y_prediction):
    """Generates a confusion matrix given ground truth and predictions

    Args:
        y_gold (np.ndarray): Ground truth/Gold Standard labels.
        y_predictions (np.ndarray): Predicted labels.
    
    Returns:
        confusion_matrix (np.ndarray): A 4 by 4 confusion matrix.
    """

    assert len(y_gold) == len(y_prediction)

    # Match labels to confusion matrix indexes.
    y_gold, y_prediction = y_gold - 1, y_prediction - 1
    
    # Populate Confusion Matrix
    confusion_matrix = np.zeros((4, 4), dtype=np.int32) # Number of rooms is 4.
    for gold, prediction in zip(y_gold, y_prediction):
        confusion_matrix[gold][prediction] += 1

    return confusion_matrix

if __name__ == "__main__":
    pass
