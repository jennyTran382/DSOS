import os.path
import scipy.io as sio
from datetime import datetime
import numpy.matlib as npm
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def sk_knn(X_train, X_test, y_train, y_test):
    k_values = [1 , 3, 5]
    scores = {}
    for k in range(len(k_values)):
        knn = KNeighborsClassifier(n_neighbors=k_values[k])
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores[k] = balanced_accuracy_score(y_test, y_pred)

        print("{},{},{},{},{}".format
                 (fold, k_values[k], scores[k], expname, str(datetime.now())))

def print_output(*argv, ee="\n"):
    print(argv[0], end=ee)
    print(argv[0], end=ee, file=f)
    f.flush()

expname = "preliminary"
all_dataset = ["mushroom_data.csv", "new_churn_data.csv", "new_spambase_data.csv", "soinn_demo_train.mat",
               "adult_data.csv", "new_icu_data.csv", "bank_data.csv", "mnist784.p"]

# INPUT
# dataset_index = 6
dataset = sys.argv[1]
max_seed = 30

# OUTPUT setting
# dataset = all_dataset[dataset_index]
data_path = "data/"
directory = dataset.split(".")[0]
if not os.path.exists("outputs/" + directory):
    os.makedirs("outputs/" + directory)

import re
# Start program
mat_file = re.compile(".*\.mat$")
csv_file = re.compile(".*\.csv$")
pickle_file = re.compile(".*\.p$")

if mat_file.match(dataset):
    input_data = sio.loadmat(data_path + dataset)
    X = input_data['train']
    y = np.zeros((X.shape[0], 1))
elif csv_file.match(dataset):
    import pandas as pd
    input_data = pd.read_csv(data_path + dataset)
    X = input_data.values[::, 1::]
    X = X.astype('float64')
    y = np.reshape(input_data.values[::, 0], (X.shape[0], 1))

elif pickle_file.match(dataset):
    import pickle
    X, y = pickle.load(open(data_path + dataset, "rb"))

filename = "outputs/" + directory + "/knn_results.csv"
if not os.path.isfile(filename):
    f = open(filename, 'a+')
    print_output("fold,k,testacc,expname,endtimestamp,runtime")
f = open(filename, 'a+')

skf = StratifiedKFold(n_splits=5)
fold = 0

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train = np.asarray([int(i) for i in y_train])
    y_test = np.asarray([int(i) for i in y_test])

    #KNN
    sk_knn(X_train, X_test, y_train, y_test)

    fold += 1
f.close()

