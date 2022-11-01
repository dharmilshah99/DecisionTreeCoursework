import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import tree
import parse
import eval

###
# Globals
###

NOISY_DATASET_PATH = "wifi_db/noisy_dataset.txt"
CLEAN_DATASET_PATH = "wifi_db/clean_dataset.txt"

###
# Main
###


def pefrorm_evaluation(path):
    """Performs Evaluation on either the Clean/Noisy Dataset.

    Args:
        path (str): Path to read the dataset from.
    """

    # Parse Dataset
    _, _, dataset = parse.read_dataset(path)

    # Evaluate
    confusion_matrix, avg_depth = eval.perform_k_fold_cross_validation(dataset, 10)
    pruned_confusion_matrix, pruned_avg_depth = eval.perform_nested_k_fold_cross_validation(dataset, 10)

    # Print Metrics
    print(create_printing_banner("*** Unpruned Tree Metrics ***"))
    eval.report_evaluation_metrics(confusion_matrix, avg_depth)

    print(create_printing_banner("*** Pruned Tree Metrics ***"))
    eval.report_evaluation_metrics(pruned_confusion_matrix, pruned_avg_depth)


def save_tree():
    """Builds and Saves Tree for the Enitre Clean Dataset"""

    # Parse Clean Dataset
    _, _, dataset = parse.read_dataset(CLEAN_DATASET_PATH)

    # Build Tree
    dtree, depth = tree.decision_tree_learning(dataset)
    tree.save_plot_tree_image(dtree, depth, "clean_dataset_tree.png")

def create_printing_banner(text, character='=', length=70):
    """Builds and Saves Tree for the Enitre Clean Dataset"""
    banner_text = ' %s ' % text
    banner = banner_text.center(length, character)
    return banner

if __name__ == "__main__":

    # Perform Evaluation
    
    print(create_printing_banner("Clean Dataset Results"))
    pefrorm_evaluation(CLEAN_DATASET_PATH)
    print(create_printing_banner("Noisy Dataset Results"))
    pefrorm_evaluation(NOISY_DATASET_PATH)

    # Plot Tree
    save_tree()
