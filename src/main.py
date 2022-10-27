import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import tree
import parse
import eval

###
# Main
###


class DecisionTree:
    def __init__(
        self,
        train_dataset=None,
        test_dataset=None,
        node=None,
        depth=None,
        accuracy=None,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.node = node
        self.depth = depth
        self.accuracy = accuracy


def parsing(d_tree: DecisionTree, path):
    x, y, dataset = parse.read_dataset(path)
    seed = 60012
    rg = default_rng(seed)
    d_tree.train_dataset, d_tree.test_dataset = parse.split_dataset(
        dataset, test_proportion=0.2, random_generator=rg
    )
    print(d_tree.train_dataset.shape)
    print(d_tree.test_dataset.shape)


def build_tree(d_tree: DecisionTree):
    d_tree.node, d_tree.depth = tree.decision_tree_learning(d_tree.train_dataset, 1)
    print(d_tree.depth)


def eval_accuracy(d_tree: DecisionTree):
    x_test = d_tree.test_dataset[:, :-1]
    y_test = d_tree.test_dataset[:, -1]
    tree_predictions = eval.predict(d_tree.node, x_test)
    d_tree.accuracy = eval.compute_accuracy_arrays(y_test, tree_predictions)
    print(d_tree.accuracy)
    return d_tree.accuracy


def save_tree(d_tree: DecisionTree):
    tree.save_plot_tree_image(d_tree.node, d_tree.depth, "tree.png")

def compute_accuracy_with_pruning():
    # split into 10 folds - each with different test set
    # for each of the remaining 9 folds 
    pass

def run_everything():
    path = "wifi_db/noisy_dataset.txt"
    _,_,dataset = parse.read_dataset(path)
    train_dataset, test_dataset = parse.split_dataset(dataset, 0.2)
    train_dataset, val_dataset = parse.split_dataset(train_dataset, 0.25)
    print(np.shape(train_dataset))
    print(np.shape(val_dataset))
    print(np.shape(test_dataset))

    d_tree = DecisionTree()
    d_tree.train_dataset, d_tree.test_dataset = train_dataset, test_dataset
    build_tree(d_tree)
    a = eval_accuracy(d_tree)
    print("count:", d_tree.node.node_count())
    tree.prune_tree(val_dataset, d_tree.node)
    b = eval_accuracy(d_tree)
    print("count:", d_tree.node.node_count())
    print("After", d_tree.node.pruned)
    print("done")
    print("Improvement: ", b - a)
    print("Accuracies: ", a, b)


def run_it():
    d_tree = DecisionTree()
    path = "wifi_db/noisy_dataset.txt"
    parsing(d_tree, path)
    build_tree(d_tree)
    eval_accuracy(d_tree)
    # save_tree(d_tree)

def run_report():
    path = "wifi_db/clean_dataset.txt"
    _, _, dataset = parse.read_dataset(path)
    eval.report_evaluation_metrics(dataset, 10)
    pass


def parse_and_build():
    d_tree = DecisionTree()
    parsing(d_tree)
    build_tree(d_tree)


if __name__ == "__main__":
    # run_it()
    # print()
    run_everything()
    # run_report()
