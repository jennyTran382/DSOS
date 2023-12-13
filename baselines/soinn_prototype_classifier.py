import somoclu
import os.path
import scipy.io as sio
import time
from datetime import datetime
import numpy.matlib as npm
import numpy as np
import soms.soinnmaster.python.fast_soinn as fast_soinn
import sys

# c1 = np.random.rand(50, 10)/5
# c2 = (0.6, 0.1, 0.05, 0.6, 0.8, 0.2, 0.5, 0.4, 0.1, 0.3) + np.random.rand(50, 10)/5
# c3 = (0.4, 0.1, 0.7, 0.03, 0.4, 0.01, 0.3, 0.2, 0.04, 0.9) + np.random.rand(50, 10)/5
# data = np.float32(np.concatenate((c1, c2, c3)))
from sklearn.model_selection import train_test_split, StratifiedKFold


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

mat_file = re.compile(".*\.mat$")
csv_file = re.compile(".*\.csv$")
pickle_file = re.compile(".*\.p$")

if mat_file.match(dataset):
    input_data = sio.loadmat(data_path + dataset)
    X = input_data['train']
    y = np.zeros((X.shape[0], 1))
    # plt.figure(1)
    # plt.plot(train[:, 0], train[:, 1], 'bo', markersize=0.5)
    # plt.show()
elif csv_file.match(dataset):
    import pandas as pd
    input_data = pd.read_csv(data_path + dataset)
    X = input_data.values[::, 1::]
    X = X.astype('float64')
    y = np.reshape(input_data.values[::, 0], (X.shape[0], 1))

elif pickle_file.match(dataset):
    import pickle
    X, y = pickle.load(open(data_path + dataset, "rb"))

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=test_set_perc, random_state=123, stratify=y)
# y_train = np.asarray([int(i) for i in y_train])
# y_test = np.asarray([int(i) for i in y_test])

n_rows, n_columns = 50, 100
filename = "outputs/" + directory + "/soinn_results.csv"
if not os.path.isfile(filename):
    f = open(filename, 'a+')
    print_output("age_max,lamb,c1,c2,seed,nbrnodes,k,trainacc,testacc,expname,endtimestamp,runtime")
f = open(filename, 'a+')

skf = StratifiedKFold(n_splits=5)
fold = 0
for train_index, test_index in skf.split(X, y):
    # print(train_index)
    # print(test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train = np.asarray([int(i) for i in y_train])
    y_test = np.asarray([int(i) for i in y_test])

    for r in range(max_seed):
        # SOINN
        start_time = time.time()
        nodes, connection, classes = fast_soinn.fast_soinn(X_train, 50, 100, 1.5, 0.001, r)    # data, age_max, lamb, c1, c2
        end_time = time.time()
        print('SOINN execute %s seconds' % (end_time - start_time))

        #treat each neuron as a prototype => assign each node with labels the same way as done with DSOS
        nbr_cls = len(np.unique(y_train))
        neuron_label = np.zeros((nodes.shape[0], nbr_cls))
        for i in range(X_train.shape[0]):
            errors_copy = (np.sum(np.square(np.matlib.repmat(X_train[i, :], nodes.shape[0], 1) - nodes), axis=1))
            closet_neuron = np.argmin(errors_copy)
            neuron_label[closet_neuron, y_train[i]] += 1
            # update also all particles connected to this particle
            # locate = np.where(connection[closet_particle, :] > 0)
            # particle_label[locate, y_train[i]] += 1
        k_values = [1, 3, 5]
        pred_train = np.zeros((y_train.shape[0], len(k_values)))
        pred_test = np.zeros((y_test.shape[0], len(k_values)))
        # Prediction on training set
        for i in range(X_train.shape[0]):
            errors = (np.sum(np.square(np.matlib.repmat(X_train[i, :], nodes.shape[0], 1) - nodes), axis=1))

            for k in range(len(k_values)):
                errors_copy = np.copy(errors)
                data_label = np.zeros((1, neuron_label.shape[1]))
                for j in range(0, k_values[k]):
                    min_error = np.amin(errors_copy)
                    closet_neuron = np.argmin(errors_copy)
                    data_label += (neuron_label[closet_neuron])  # / min_error)
                    errors_copy[closet_neuron] = 10000
                pred_train[i][k] = np.argmax(data_label)
        # Prediction on test set
        for i in range(X_test.shape[0]):
            errors = (np.sum(np.square(np.matlib.repmat(X_test[i, :], nodes.shape[0], 1) - nodes), axis=1))

            for k in range(len(k_values)):
                errors_copy = np.copy(errors)
                data_label = np.zeros((1, neuron_label.shape[1]))
                for j in range(0, k_values[k]):
                    min_error = np.amin(errors_copy)
                    closet_neuron = np.argmin(errors_copy)
                    data_label += (neuron_label[closet_neuron])  # / min_error)
                    errors_copy[closet_neuron] = 10000
                pred_test[i][k] = np.argmax(data_label)

        from sklearn.metrics import balanced_accuracy_score  # , accuracy_score, f1_score

        for k in range(len(k_values)):
            train_acc = balanced_accuracy_score(y_train, pred_train[:, k])
            test_acc = balanced_accuracy_score(y_test, pred_test[:, k])
            # print(test_acc, accuracy_score(y_test, pred_test), f1_score(y_test, pred_test))

            print_output("{},{},{},{},{},{},{},{},{},{},{},{}".format
                         (50, 100, 1.5, 0.001, r, fold, nodes.shape[0], k_values[k], train_acc, test_acc, expname, str(datetime.now()),
                          end_time - start_time))
    fold += 1
f.close()

