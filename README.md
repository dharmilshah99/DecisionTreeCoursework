# DecisionTreeCoursework
Introduction to ML - Decision Tree Coursework
The purpose of the coursework is to implement a decision tree algorithm and use it to determine one of the indoor
locations based on WIFI signal strengths collected from a mobile phone. The decicion tree should be built and tested using the two datasets provided: `wifi_db/clean_dataset.txt` and `wifi_db/noisy_dataset.txt`.
The coursework can be split into 4 major steps:
1. Loading Data
2. Creating Decision Trees
3. Evaluation of trees using 10-fold cross validation
4. Pruning and analysis of the trees


## Setting up Virtual Environment

A virtual environment should be setup before installing packages in prerequisites. Python virtual environment can be activated via:
```sh
cd your_project_directory
python3 -m venv venv
source venv/bin/activate
```

## Prerequisites

The packages in `requirements.txt` should be installed by running:
```sh
pip install --upgrade pip
pip install -r requirements.txt
python3 -c "import numpy as np; import torch; print(np); print(torch)"
```

## Running the Program

Run `src/main.py` from the root directory to
- Create pruned and unpruned decision trees for each of the clean and noisy dataset
- Print out evaluation metrics for each tree
- Saves a visualisation of each tree in `images`

The program can be ran with the following command:
```sh
python3 src/main.py
```

## Output

After running the program, the following data are being printed for both `wifi_db/clean_dataset.txt` and `wifi_db/noisy_dataset.txt`:
1. Average Tree Depth
2. Average of Confusion Matrices over 10 folds
3. Average Overall Accuracy
4. Precision
5. Recall
6. F1-Score
