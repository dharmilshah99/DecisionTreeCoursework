import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import tree
import parse
import eval

###
# Main
###

if __name__ == "__main__":
    # Parse
    path = "wifi_db/clean_dataset.txt"
    x, y, dataset = parse.read_dataset(path)

    seed = 60012
    rg = default_rng(seed)
    train_dataset, test_dataset = parse.split_dataset(
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
    tree.save_plot_tree_image(node, depth, "tree.png")
