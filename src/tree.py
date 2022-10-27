import numpy as np
from matplotlib import pyplot as plt

import node
import eval

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
        return node.Node(label=dataset[:, -1][0]), depth
    else:
        # Split
        attribute, split = find_split(dataset)
        l_dataset = dataset[dataset[:, attribute] < split]
        r_dataset = dataset[dataset[:, attribute] >= split]

        # Create Node
        left, l_depth = decision_tree_learning(l_dataset, depth + 1)
        right, r_depth = decision_tree_learning(r_dataset, depth + 1)

        return node.Node(left, right, attribute, split), max(l_depth, r_depth)


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


def plot_tree(node, depth, width, x=0, y=0):
    """Recursively Plots Decision Tree.

    Args:
        node (Node): Root node of the decision tree.
        depth (int): Maximum depth of the decision tree.
        width (int): Width of the image for the tree to be plotted on.
        x (int, optional): X Coordinate to plot the root node.
        y (int, optional): Y Coordinate to plot the leaf node.
    """

    if node.is_leaf():
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

def prune_tree(dataset, root):
    """Prunes tree to improve accuracy

    Args:
        dataset (np.ndarray): 
        root (Node): 

    Returns:
        None: modifies root in place
    """
    root.pruned = True
    # Terminating Conditions
    if root.is_leaf(): 
        return

    #TODO: is this correct? Added this in to fix zero len bug, is it good? Dont prune if there's no validation data to decide on
    # Merge with top condition once verified
    if len(dataset) == 0:
        return 

    # print(f"x_{root.attribute} < {root.value}")
    prune_tree(dataset[dataset[:, root.attribute] < root.value], root.left)
    prune_tree(dataset[dataset[:, root.attribute] >= root.value], root.right)
    
    # Check Accuracy
    if root.left.is_leaf() and root.right.is_leaf(): 
        # Compute Accuracy on Val
        y_gold, y_prediction = dataset[:, -1], eval.predict(root, dataset[:, :-1])

        accuracy = eval.compute_accuracy_arrays(y_gold, y_prediction)

        # Get the majority label and count, decide whether to replace
        values, counts = np.unique(dataset[:, -1], return_counts=True)

        # # getting the majority label of validation set without considering the labels of the root nodes: not correct? 
        # majority_label = values[np.argmax(counts)]
        # majority_count = np.amax(counts)

        #TODO is this the correct? IDK but it seems to work a bit better than above
        num_from_val = dict(zip(values, counts))
        tmp_list = [(root.left.label, num_from_val.get(root.left.label,0)), (root.right.label, num_from_val.get(root.right.label,0))]
        majority_label, majority_count = max(tmp_list, key=lambda item:item[1])

        print(f"Accuracies: {accuracy}, {majority_count / len(dataset)}")
        if accuracy <= majority_count / len(dataset):
            # this doesnt work... root is a reference to a Python object, doing this simply points root to a different object leaving the original unchanged
            # root = node.Node(label=majority_label)

            root.make_leaf(majority_label) # probs more pythonic way of doing this...
            print("Pruned")

    return

if __name__ == "__main__":
    pass
