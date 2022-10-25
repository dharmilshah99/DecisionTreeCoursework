from numpy.random import default_rng
import numpy as np

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
    n_test = round(len(dataset) * test_proportion)
    n_train = len(dataset) - n_test
    train_dataset = dataset[shuffled_indices[:n_train]]
    test_dataset = dataset[shuffled_indices[n_train:]]
    return (train_dataset, test_dataset)


if __name__ == "__main__":
    pass