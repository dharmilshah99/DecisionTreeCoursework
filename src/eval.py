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
    y_gold_idx, y_prediction_idx = y_gold - 1, y_prediction - 1

    # Populate Confusion Matrix
    confusion_matrix = np.zeros((4, 4), dtype=np.int32)  # Number of rooms is 4.
    for gold, prediction in zip(y_gold_idx, y_prediction_idx):
        confusion_matrix[gold][prediction] += 1

    return confusion_matrix


def compute_accuracy(confusion_matrix):
    """Compute the accuracy given a confusion matrix.

    Args:
        conufusion_matrix (np.ndarray): A 4 by 4 confusion matrix.

    Returns:
        float: Accuracy.
    """

    try:
        return np.trace(confusion_matrix) / confusion_matrix.sum()
    except ZeroDivisionError:
        return 0


def compute_precision(confusion_matrix):
    """Compute the precision score per class given the ground truth and predictions

    Also return the macro-averaged precision across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the
              precision for class c
            - macro-precision is macro-averaged precision (a float)
    """
    p = np.zeros((len(confusion_matrix),))

    for i in range(len(confusion_matrix)):
        total = np.sum(confusion_matrix[:, i])
        if total > 0:
            p = confusion_matrix[i][i] / total

    # Compute the macro-averaged precision
    macro_p = 0
    if len(p) > 0:
        macro_p = np.mean(p)
    return (p, macro_p)


def compute_recall(confusion_matrix):
    """Compute the recall score per class given the ground truth and predictions

    Also return the macro-averaged recall across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the
                recall for class c
            - macro-recall is macro-averaged recall (a float)
    """
    r = np.zeros((len(confusion_matrix),))

    for i in range(len(confusion_matrix)):
        total = np.sum(confusion_matrix[i, :])
        if total > 0:
            r = confusion_matrix[i][i] / total

    # Compute the macro-averaged recall
    macro_r = 0
    if len(r) > 0:
        macro_r = np.mean(r)
    return (r, macro_r)


def compute_f1_score(y_gold, y_prediction):
    """Compute the F1-score per class given the ground truth and predictions

    Also return the macro-averaged F1-score across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float)
    """

    (precisions, macro_p) = compute_precision(y_gold, y_prediction)
    (recalls, macro_r) = compute_recall(y_gold, y_prediction)

    # Just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions),))
    f = 2 * precisions * recalls / (precisions + recalls)

    macro_f = 0
    if len(f) > 0:
        macro_f = np.mean(f)
    return (f, macro_f)


if __name__ == "__main__":
    pass
