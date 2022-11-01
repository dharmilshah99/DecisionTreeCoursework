import numpy as np
from numpy.random import default_rng

import tree


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
        while not curr_node.is_leaf():
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
    # Number of rooms is 4.
    confusion_matrix = np.zeros((4, 4), dtype=np.int32)
    for gold, prediction in zip(y_gold_idx, y_prediction_idx):
        confusion_matrix[int(gold)][int(prediction)] += 1

    return confusion_matrix


def compute_accuracy_arrays(y_gold, y_prediction):
    """Compute the accuracy given the ground truth and predictions

    Args:
    y_gold (np.ndarray): the correct ground truth/gold standard labels
    y_prediction (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_gold) == len(y_prediction)

    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0


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
        conufusion_matrix (np.ndarray): A 4 by 4 confusion matrix.

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
            p[i] = confusion_matrix[i][i] / total

    # Compute the macro-averaged precision
    macro_p = 0
    if len(p) > 0:
        macro_p = np.mean(p)
    return (p, macro_p)


def compute_recall(confusion_matrix):
    """Compute the recall score per class given the ground truth and predictions

    Also return the macro-averaged recall across classes.

    Args:
        conufusion_matrix (np.ndarray): A 4 by 4 confusion matrix.

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
            r[i] = confusion_matrix[i][i] / total

    # Compute the macro-averaged recall
    macro_r = 0
    if len(r) > 0:
        macro_r = np.mean(r)
    return (r, macro_r)


def compute_f1_score(confusion_matrix):
    """Compute the F1-score per class given the ground truth and predictions

    Also return the macro-averaged F1-score across classes.

    Args:
        conufusion_matrix (np.ndarray): A 4 by 4 confusion matrix.

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float)
    """

    (precisions, _) = compute_precision(confusion_matrix)
    (recalls, _) = compute_recall(confusion_matrix)

    # Just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions),))
    f = 2 * precisions * recalls / (precisions + recalls)

    macro_f = 0
    if len(f) > 0:
        macro_f = np.mean(f)
    return (f, macro_f)


def k_fold_split(n_splits, dataset, random_generator=default_rng()):
    """Split dataset into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        dataset (int): The dataset to split up
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """
    # Shuffle and split dataset into n_splits
    shuffle = random_generator.permutation(len(dataset))
    dataset = dataset[shuffle]
    dataset_splits = np.array_split(dataset, n_splits)

    return dataset_splits


def perform_k_fold_cross_validation(
    dataset, n_splits=10, random_generator=default_rng()
):
    """Performs K-Fold Cross Validation

    Args:
        dataset (np.ndarray): Instances, numpy array with shape (N,K+1).
        n_splits (int): Number of splits. Defaults to 10.
        random_generator (np.random.Generator): A numpy random generator.

    Returns:
        Average Confusion Matrix (np.array): Average 4 by 4 confusion matrix over all folds.
    """

    # Shuffle and Split Dataset
    dataset_splits = k_fold_split(n_splits, dataset, random_generator)

    # Run K-Fold Cross Validation
    confusion_matrices, depths = np.zeros((n_splits, 4, 4)), np.zeros((n_splits, 1))
    for i in range(n_splits):
        # Split
        test_dataset = dataset_splits[i]
        train_dataset = np.vstack(dataset_splits[:i] + dataset_splits[i + 1 :])

        # Train
        dtree, depth = tree.decision_tree_learning(train_dataset, 1)

        # Evaluate
        y_gold, y_prediction = test_dataset[:, -1], predict(dtree, test_dataset[:, :-1])

        # Save Depth and Confusion Matrix for a Fold
        depths[i] = depth
        confusion_matrices[i] = generate_confusion_matrix(y_gold, y_prediction)

    return np.mean(confusion_matrices, axis=0), np.mean(depths)


def perform_nested_k_fold_cross_validation(
    dataset, n_splits=10, random_generator=default_rng()
):
    """Performs nested K-Fold Cross Validation

    Args:
        dataset (np.ndarray): Instances, numpy array with shape (N,K+1).
        n_splits (int): Number of splits. Defaults to 10.
        random_generator (np.random.Generator): A numpy random generator.

    Returns:
        Average Confusion Matrix (np.array): Average 4 by 4 confusion matrix over all folds.
    """

    # Shuffle and Split Dataset
    dataset_splits = k_fold_split(n_splits, dataset, random_generator)

    # Run Nested K-Fold Cross Validation
    confusion_matrices = np.zeros((n_splits, 4, 4))
    depths = np.zeros((n_splits, 1))

    for i in range(n_splits):

        # Split into Test and Train + Validation Datasets
        test_dataset = dataset_splits[i]
        train_validation_dataset = dataset_splits[:i] + dataset_splits[i + 1 :]

        # Perform Nested Validation
        best_confusion_matrix, best_accuracy, best_depth = None, 0, 0
        for j in range(n_splits - 1):

            # Split into Train and Validation Datasets
            validation_dataset = train_validation_dataset[j]
            train_dataset = np.vstack(
                train_validation_dataset[:j] + train_validation_dataset[j + 1 :]
            )

            # Train & Prune
            dtree, depth = tree.decision_tree_learning(train_dataset, 1)
            tree.prune_tree(validation_dataset, dtree)
            pruned_depth = dtree.get_depth()
            assert pruned_depth <= depth

            # Evaluate
            y_gold = test_dataset[:, -1]
            y_prediction = predict(dtree, test_dataset[:, :-1])

            # Keep Track of Best Tree
            confusion_matrix = generate_confusion_matrix(y_gold, y_prediction)
            accuracy = compute_accuracy(confusion_matrix)

            if compute_accuracy(confusion_matrix) > best_accuracy:
                best_confusion_matrix = confusion_matrix
                best_accuracy = accuracy
                best_depth = pruned_depth

        # Keep Track of Confusion Matrices
        confusion_matrices[i] = best_confusion_matrix
        depths[i] = best_depth

    return np.mean(confusion_matrices, axis=0), np.mean(depths)


def report_evaluation_metrics(confusion_matrix, avg_depth, n_splits=10):
    """Reports accuracy, precision and recall rates per class, and F1 measures.

    Args:
        dataset (np.ndarray): Instances, numpy array with shape (N,K+1).
        avg_depth (float): Average tree depth.
        n_splits (int): Number of splits. Defaults to 10.
    """

    print(f"Average Tree Depth:\n {avg_depth}\n")
    print(f"Average Confusion Matrix over {n_splits} folds:\n {confusion_matrix}\n")

    avg_accuracy = compute_accuracy(confusion_matrix)
    print(f"Average Overall Accuracy: {avg_accuracy}\n")

    class_precisions, macro_precision = compute_precision(confusion_matrix)
    print(f"Precision per Class: {class_precisions}")
    print(f"Macro Precision: {macro_precision}\n")

    class_recalls, macro_recall = compute_recall(confusion_matrix)
    print(f"Recalls per Class: {class_recalls}")
    print(f"Macro Recall: {macro_recall}\n")

    class_f_score, macro_f = compute_f1_score(confusion_matrix)
    print(f"F1-Score per Class: {class_f_score}")
    print(f"Macro F1-Score: {macro_f}\n")


if __name__ == "__main__":
    pass
