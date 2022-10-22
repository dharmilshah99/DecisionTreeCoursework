import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import tree
import eval

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


def plot_tree(node, depth, width, x=0, y=0):
    """Recursively Plots Decision Tree.

    Args:
        node (Node): Root node of the decision tree.
        depth (int): Maximum depth of the decision tree.
        width (int): Width of the image for the tree to be plotted on.
        x (int, optional): X Coordinate to plot the root node.
        y (int, optional): Y Coordinate to plot the leaf node.
    """

    if node.is_root():
        # Plot Leaf
        plt.text(
            x,
            y,
            r"${}$".format(str(node.label)),
            color="black",
            bbox=dict(facecolor="white", edgecolor="green", boxstyle="round,pad=1"),
        )
        return
    else:
        # Plot Decision
        text = f"X_{node.attribute} < {node.value}"
        plt.text(
            x,
            y,
            r"${}$".format(text),
            color="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=1"),
        )

        # Plot Edges
        y_l = y_r = y - 1
        x_l, x_r = x - width / 2, x + width / 2
        plt.plot([x_l, x, x_r], [y_l, y, y_r])

        # Recurse
        plot_tree(node.left, depth - 1, width / 2, x_l, y_l)
        plot_tree(node.right, depth - 1, width / 2, x_r, y_r)
        return


def save_plot_tree_image(node, depth, filename):
    """Saves an image of the tree plot.

    Args:
        node (Node): Root node of the decision tree.
        depth (int): Maximum depth of the decision tree.
        filename (str): Filename to save image under the "images/" folder.
    """
    plt.figure(figsize=(128, 128), dpi=100)  # TODO: Check image size limits.
    plot_tree(node, depth, 128)
    plt.savefig("images/" + filename)


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
    node, depth = tree.decision_tree_learning(train_dataset, 1)
    print(depth)

    # Evaluate Accuracy
    x_test = test_dataset[:, :-1]
    y_test = test_dataset[:, -1]
    tree_predictions = eval.predict(node, x_test)
    accuracy = eval.compute_accuracy(y_test, tree_predictions)
    print(accuracy)

    # Save Tree
    save_plot_tree_image(node, depth, "tree.png")
