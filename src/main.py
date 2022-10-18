import numpy as np


###
# Objects
###


class Node:
    def __init__(self, left=None, right=None, attribute=None, value=None):
        self.left = left
        self.right = right
        self.attribute = attribute
        self.value = value


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
            row = line.split("\t")
            x.append(list(map(float, row[:-1])))
            y.append(int(row[-1]))
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
        return Node(), depth
    else:
        # Split
        attribute, split = find_split(dataset)
        l_dataset = dataset[dataset[:, attribute] < split]
        r_dataset = dataset[dataset[:, attribute] >= split]

        # Create Node
        left, l_depth = decision_tree_learning(l_dataset, depth + 1)
        right, r_depth = decision_tree_learning(r_dataset, depth + 1)

        return Node(left, right, attribute, split), max(l_depth, r_depth)


###
# Main
###

if __name__ == "__main__":
    # Parse
    path = "wifi_db/clean_dataset.txt"
    x, y, dataset = read_dataset(path)

    # Build Tree
    node, depth = decision_tree_learning(dataset, 1)
    print(depth)
