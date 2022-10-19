from re import A
import numpy as np
from numpy.random import default_rng

###
# Objects
###


class Node:
    """Represents a Decision Tree Node."""

    def __init__(self, left=None, right=None, attribute=None, value=None, label=None):
        """Initializes Decision Tree Node.

        Args:
            left (Node, optional): Node left of current node. Defaults to None.
            right (Node, optional): Node right of current node. Defaults to None.
            attribute (int, optional): Attribute to split dataset upon. Defaults to None.
            value (float, optional): Split point value. Defaults to None.
            label (int, optional): Classification label of Node. Defaults to None.
        """
        self.left = left
        self.right = right
        self.attribute = attribute
        self.value = value
        self.label = label

    def is_root(self):
        """Checks if Node is a root.

        Returns:
            True if Node has children. False, otherwise.
        """
        return (self.left == None) and (self.right == None)


###
# Helpers
###


def read_dataset(path):
    """Reads dataset from specified path.

    Args:
        path (str): Path to read the dataset from.

    Returns:
        tuple: Returns a tuple (x, y, z) of numpy arrays.
            - x: A numpy array with shape (N, K) where N is the number of instances and K is the number of features.
            - y: A numpy array with shape (N, ).
            - z: A numpy array with shape (N, K+1). Includes the labels in the last column.
    """
    x, y = [], []
    for line in open(path):
        line = line.strip()
        if line:
            row = line.split()
            x.append(list(map(float, row[:-1])))
            y.append(int(float(row[-1])))
    x = np.array(x)
    y = np.array(y)
    return x, y, np.column_stack((x, y))


def get_entropy(labels, total):
    """Computes entropy given a dictionary containing counts for each label and the total number of all labels.

    Args:
        labels (dict): Dict containing counts for each label.
        total (int): Sum of counts across all labels.

    Returns:
        h (float): Returns information entropy given Y.
    """
    h = 0
    for count in labels.values():
        if count != 0:
            pk = float(count) / total
            h -= pk * np.log2(pk)
    return h


def find_split(dataset):
    """Chooses value in X to split upon that results in the highest information gain.

    Args:
        dataset (np.ndarray): Numpy array with shape (N, K+1). Includes K attributes and 1 label.

    Returns:
        Tuple: Returns a tuple of (feature, split).
            - feature (int): Feature that produced the maximum information gain.
            - split (float): Split upon feature.
    """
    x, y = dataset[:, :-1], dataset[:, -1]

    # Returns
    max_gain = 0
    split = feature = None

    # Iterate over Features
    features = x.shape[1]
    for i in range(features):

        # Sort on Feature
        p = np.argsort(x[:, i])
        x_sorted, y_sorted = x[p], y[p]

        # Initialize running totals for each label
        left, right = (
            {label: 0 for label in np.unique(y)},
            {label: np.count_nonzero(y == label) for label in np.unique(y)},
        )
        s_left = 0
        s_right = total = sum(right.values())

        # Find overall entropy
        h_all = get_entropy(right, total)

        # Find optimal split point For a feature
        previous_val = None
        for idx, val in enumerate(x_sorted[:, i]):
            # Check for change in val
            if previous_val != val:
                # Compute gain
                h_left, h_right = get_entropy(left, s_left), get_entropy(right, s_right)
                remainder = (h_left * s_left / total) + (h_right * s_right / total)
                gain = h_all - remainder

                # Keep track of maximum gain
                if gain >= max_gain:
                    max_gain = gain
                    split = val
                    feature = i

            # Update running totals
            s_left += 1
            s_right -= 1
            left[y_sorted[idx]] += 1
            right[y_sorted[idx]] -= 1

            previous_val = val

    return (feature, split)


def decision_tree_learning(dataset, depth):
    """Builds decision tree recusively.

    Args:
        dataset (np.ndarray): Numpy array with shape (N, K+1). Includes K attributes and 1 label.
        depth (int): Layers of decision tree to build

    Returns:
        Tuple: Returns a tuple of (feature, depth).
            - feature (int): Feature that produced the maximum information gain.
            - depth (float): Split upon feature.
    """
    # Terminating Condition
    if np.all(dataset[:, -1] == dataset[:, -1][0]):
        return Node(label=dataset[:, -1][0]), depth
    else:
        # Split
        attribute, split = find_split(dataset)
        l_dataset = dataset[dataset[:, attribute] < split]
        r_dataset = dataset[dataset[:, attribute] >= split]

        # Create Node
        left, l_depth = decision_tree_learning(l_dataset, depth + 1)
        right, r_depth = decision_tree_learning(r_dataset, depth + 1)

        return Node(left, right, attribute, split), max(l_depth, r_depth)


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


def split_dataset(dataset, test_proportion, random_generator=default_rng()):
    """ Split dataset into training and test sets, according to the given test set proportion.

    Args:
        dataset (np.ndarray): Instances, numpy array with shape (N,K+1).
        test_proprotion (float): the desired proportion of test examples (0.0-1.0)
        random_generator (np.random.Generator): A random generator

    Returns:
        tuple: returns a tuple of (train_dataset, test_dataset) 
            - train_dataset (np.ndarray): Training dataset shape (N_train, K+1)
            - test_dataset (np.ndarray): Test instances shape (N_test, K+1)
    """

    shuffled_indices = random_generator.permutation(len(dataset))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    train_dataset = dataset[shuffled_indices[:n_train]]
    test_dataset = dataset[shuffled_indices[n_train:]]
    return (train_dataset, test_dataset)


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

###
# Main
###


if __name__ == "__main__":
    # Parse
    path = "wifi_db/clean_dataset.txt"
    x, y, dataset = read_dataset(path)

    seed = 60012
    rg = default_rng(seed)
    train_dataset, test_dataset = split_dataset(
        dataset, test_proportion=0.2, random_generator=rg
    )
    print(train_dataset.shape)
    print(test_dataset.shape)

    # Build Tree
    node, depth = decision_tree_learning(train_dataset, 1)
    print(depth)

    # Evaluate Accuracy
    x_test = test_dataset[:, :-1]
    y_test = test_dataset[:, -1]
    tree_predictions = predict(node, x_test)
    accuracy = compute_accuracy(y_test, tree_predictions)
    print(accuracy)
