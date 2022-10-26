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
    d_tree.accuracy = eval.compute_accuracy(y_test, tree_predictions)
    print(d_tree.accuracy)


def save_tree(d_tree: DecisionTree):
    tree.save_plot_tree_image(d_tree.node, d_tree.depth, "tree.png")


def run_everything():
    d_tree = DecisionTree()
    path = "wifi_db/clean_dataset.txt"
    parsing(d_tree, path)
    build_tree(d_tree)
    eval_accuracy(d_tree)
    save_tree(d_tree)


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
    # run_everything()
    run_report()
