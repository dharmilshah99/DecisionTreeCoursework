import numpy as np

###
# Helpers
###


def read_dataset(path):
    """Reads dataset from specified path.

    Args:
        path (str): Path to read the dataset from.

    Returns:
        tuple: Returns a tuple (x, y) of numpy arrays.
            - x: A numpy array with shape (N, K) where N is the number of instances and K is the number of features.
            - y: A numpy array with shape (N, ).
    """
    x, y = [], []
    for line in open(path):
        line = line.strip()
        if line:
            row = line.split("\t")
            x.append(list(map(float, row[:-1])))
            y.append(int(row[-1]))
    return np.array(x), np.array(y)


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


# TODO: Verify this works
def find_split(x, y):
    """Chooses value in X to split upon that results in the highest information gain.

    Args:
        x (np.ndarray): Numpy array with shape (N, K) where N is the number of instances and K is the number of features.
        y (np.ndarray): Numpy array of shape (N,). Represents class labels.

    Returns:
        Tuple: Returns a tuple of (feature, split).
            - feature (int): Feature that produced the maximum information gain.
            - Split (float): Split upon feature.
    """

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
        for idx, val in enumerate(x_sorted[1:, i]):

            # Update running totals
            s_left += 1
            s_right -= 1
            left[y_sorted[idx]] += 1
            right[y_sorted[idx]] -= 1

            # Compute gain
            h_left, h_right = get_entropy(
                left, s_left), get_entropy(right, s_right)
            remainder = (h_left * s_left / total) + (h_right * s_right / total)
            gain = h_all - remainder

            # Keep track of maximum gain
            if gain >= max_gain:
                max_gain = gain
                split = val
                feature = i

    return (feature, split)

###
# Main
###


if __name__ == "__main__":
    # Parse
    path = "wifi_db/clean_dataset.txt"
    x, y = read_dataset(path)
    print(x[:5])
    print(y[:5])
    # Compute Split
    feature, split = find_split(x, y)
    print(feature)
    print(split)
