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


def get_entropy(y):
    """Computes entropy given Y.

    Args:
        y (np.ndarray): Numpy array of shape (N,). Represents class labels.

    Returns:
        H (float): Returns information entropy given Y.
    """
    h = 0
    labels = np.unique(y)
    for label in labels:
        pk = float(np.count_nonzero(y == label)) / len(y)
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
    split = None
    feature = None

    # Iterate over Features
    _, features = np.shape(x)
    for i in range(features):

        # Sort on Feature
        p = np.argsort(x[:, i])
        x_sorted, y_sorted = x[p], y[p]

        # Find Overall Entropy
        h_all = get_entropy(y_sorted)

        # Find Optimal Split point For a Feature
        for idx, val in enumerate(x_sorted[:, i]):
            # Compute Gain
            s_left, s_right = y_sorted[:idx], y_sorted[idx:]
            h_left, h_right = get_entropy(s_left), get_entropy(s_right)
            remainder = (h_left * len(s_left) / (len(y_sorted))) + (
                h_right * len(s_right) / (len(y_sorted))
            )
            gain = h_all - remainder

            # Keep Track of Maximum Gain
            if gain >= max_gain:
                max_gain = gain
                split = val
                feature = i

    return (feature, split)


###
# Main
###

if __name__ == "__main__":
    path = "wifi_db/clean_dataset.txt"
    x, y = read_dataset(path)
    print(x[:5])
    print(y[:5])
