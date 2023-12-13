__author__ = 'sunguyen'

import time
# import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from dsos import dsos as dsos
import numpy as np
import os.path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import sys

def print_output(*argv, ee="\n"):
    print(argv[0], end=ee)
    print(argv[0], end=ee, file=f)
    f.flush()


# seed = 1
# resol = 0.1
# alpha = 0.05
lamb = 1000
expname = "preliminary"
#all_dataset = ["mushroom_data.csv", "new_churn_data.csv", "new_spambase_data.csv", "soinn_demo_train.mat",
               # "adult_data.csv", "new_icu_data.csv", "bank_data.csv", "mnist784.p"]

# INPUT
dataset = sys.argv[1]
epoch = int(sys.argv[2])
max_seed = 30
# test_set_perc = 0.2

# OUTPUT setting
data_path = "data/"
directory = dataset.split(".")[0]
if not os.path.exists("outputs/" + directory):
    os.makedirs("outputs/" + directory)
if not os.path.exists("graphics/" + directory):
    os.makedirs("graphics/" + directory)
filename = "outputs/" + directory + "/dsos_results.csv"
if not os.path.isfile(filename):
    f = open(filename, 'a+')
    print_output("lambda,resol,alpha,seed,fold,epoch,swarmsize,meanerror,k,trainacc,testacc,expname,endtimestamp,runtime")
f = open(filename, 'a+')

print("DSOS on ", dataset)

def plot_soinn(nodes, connection, title=""):
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    for i in range(0, nodes.shape[0]):
        for j in range(0, nodes.shape[0]):
            if connection[i, j] != 0:
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-', alpha=0.3)
                pass
    plt.title(title)
    plt.show()


# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / float(N)


### Running mean/Moving average
def running_mean(l, N):
    sum = 0
    result = list(0 for x in l)

    for i in range(0, N):
        sum = sum + l[i]
        result[i] = sum / (i + 1)

    for i in range(N, len(l)):
        sum = sum - l[i - N] + l[i]
        result[i] = sum / N

    return np.array(result)


# x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# y = running_mean(x, 4)
import re

mat_file = re.compile(".*\.mat$")
csv_file = re.compile(".*\.csv$")
pickle_file = re.compile(".*\.p$")

alpha = 0.00
# plt.switch_backend('Qt4Agg')

# Read data
start_time = time.time()
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
    X, y = pickle.load(open(data_path + dataset, "rb"))
end_time = time.time()
print('loading train data executes %s seconds' % (end_time - start_time))

skf = StratifiedKFold(n_splits=5)
fold = 0
for train_index, test_index in skf.split(X, y):
    # print(train_index)
    # print(test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train = np.asarray([int(i) for i in y_train])
    y_test = np.asarray([int(i) for i in y_test])

    for resol in [0.05, 0.1]:  # , 0.2, 0.3, 0.4]:
        for seed in range(max_seed):  # [1]: #
            param = "resol{}_alpha{}_lamb{}_seed{}_fold{}_epoch{}".format(resol, alpha, lamb, seed, fold, epoch)
            np.random.seed(seed)
            start_time = time.time()

            dsos_output_file = "outputs/{}/dsos_out_{}.pickle".format(directory, param)
            if not os.path.isfile(dsos_output_file):
                particles, connection, Z, energy, trace_energy, trace_error = \
                    dsos.learning(input_data=X_train, max_nepoch=epoch, spread_factor=resol, lamb=lamb, alpha=alpha)
                # Save DSOS output
                dsos_out = (particles, connection, Z, energy, trace_energy, trace_error)
                pickle_out = open(dsos_output_file, "wb")
                pickle.dump(dsos_out, pickle_out)
                pickle_out.close()
            else:
                with open(dsos_output_file, 'rb') as pickle_file:
                    particles, connection, Z, energy, trace_energy, trace_error = pickle.load(pickle_file)

            end_time = time.time()
            print('dsos execute %s seconds' % (end_time - start_time))

            total_error = 0
            for i in range(X_train.shape[0]):
                errors = (np.sum(np.square(np.matlib.repmat(X_train[i, :], particles.shape[0], 1) - particles), axis=1))
                min_error = np.amin(errors)
                total_error += min_error

            if dataset == "soinn_demo_train.mat":  # "circles":
                plot_soinn(particles, connection, "Number of nodes = {}".format(particles.shape[0]))
                # plot_soinn(Z, connection)
                plt.scatter(particles[::, 0], particles[::, 1], s=1 + np.round(20 * energy),
                            c=["red" if x else "blue" for x in (energy == 0)])

                plt.show()

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax2.plot(np.array(trace_energy)[::, 1], label="Mean Energy", color='blue', linewidth=0.5, alpha=0.1)
            ax1.plot(np.array(trace_energy)[::, 2], label="Swarm Size", color="red", linewidth=0.5, alpha=0.7)
            ax1.plot(np.array(trace_energy)[::, 0], label="Total Energy", color="green", linewidth=0.5, alpha=0.7)
            ax1.legend()
            ax2.legend()
            ax1.set_axisbelow(True)
            ax1.yaxis.grid(color='gray', linestyle='dashed')
            plt.title("Number of nodes = " + str(particles.shape[0]))
            plt.show()

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            lag = int(len(trace_error) * 0.2)
            ax1.plot(np.array(trace_error)[lag:], label="Error", color="gray", linewidth=0.5, alpha=0.3)
            ax2.plot(running_mean(np.array(trace_error), lag)[lag:], label="Moving Average Error", color="red",
                     linewidth=0.7)
            ax1.legend()
            ax2.legend()
            ax2.yaxis.grid(color='green', alpha=0.5, linestyle='dashed')
            plt.title("Mean Error = {}".format(total_error / X_train.shape[0]))

            plt.savefig("graphics/{}/error_curve_{}.png".format(directory, param))
            plt.show()

            # import umap
            # import warnings
            #
            # warnings.simplefilter('ignore')
            #
            # reducer = umap.UMAP(n_components=2, spread=3)
            # Z = reducer.fit_transform(particles)
            #
            # for _ in range(3):
            # #while True:
            #     zdist = []
            #     zindex = []
            #     for i in range(Z.shape[0]):
            #         for j in range(Z.shape[0]):
            #             if j > i and connection[i, j] == 1:
            #                 dist = np.linalg.norm(Z[i] - Z[j])
            #                 zdist.append(dist)
            #                 zindex.append((i, j))
            #
            #     plot_soinn(Z, connection, "UMAP of DSOS")
            #     mean_zdist = np.mean(zdist)
            #     std_zdist = np.std(zdist)
            #     zdist = (np.array(zdist) - mean_zdist) / std_zdist
            #
            #     cut_count = 0
            #     for idx in range(len(zindex)):
            #         i, j = zindex[idx][0], zindex[idx][1]
            #         if zdist[idx] > 3:
            #             # print(zdist[idx], ">>", 3)
            #             connection[i, j] = 0
            #             connection[j, i] = 0
            #             cut_count += 1
            #
            #     plot_soinn(Z, connection, "UMAP of DSOS simplified")
            #     if dataset == "circles":
            #         plot_soinn(particles, connection, "DSOS simplified")
            #
            #     #if cut_count == 0:
            #     print("Cut ", cut_count, " connections.")
            #         # break

            # Binh add: evaluating DSOS as a nearest neighbour classifier
            nbr_cls = len(np.unique(y_train))
            particle_label = np.zeros((particles.shape[0], nbr_cls))
            for i in range(X_train.shape[0]):
                errors = (np.sum(np.square(np.matlib.repmat(X_train[i, :], particles.shape[0], 1) - particles), axis=1))
                closet_particle = np.argmin(errors)
                particle_label[closet_particle, y_train[i]] += 1
                # update also all particles connected to this particle
                # locate = np.where(connection[closet_particle, :] > 0)
                # particle_label[locate, y_train[i]] += 1

            k_values = [1, 3, 5]
            pred_train = np.zeros((y_train.shape[0], len(k_values)))
            pred_test = np.zeros((y_test.shape[0], len(k_values)))
            # Prediction on training set
            for i in range(X_train.shape[0]):
                errors = (
                    np.sum(np.square(np.matlib.repmat(X_train[i, :], particles.shape[0], 1) - particles),
                           axis=1))

                for k in range(len(k_values)):
                    errors_copy = np.copy(errors)
                    data_label = np.zeros((1, particle_label.shape[1]))
                    for j in range(0, k_values[k]):
                        min_error = np.amin(errors_copy)
                        best_match = np.argmin(errors_copy)
                        data_label += (particle_label[best_match])  # / min_error)
                        errors_copy[best_match] = 10000
                    pred_train[i][k] = np.argmax(data_label)
            # Prediction on test set
            for i in range(X_test.shape[0]):
                errors = (
                    np.sum(np.square(np.matlib.repmat(X_test[i, :], particles.shape[0], 1) - particles),
                           axis=1))

                for k in range(len(k_values)):
                    errors_copy = np.copy(errors)
                    data_label = np.zeros((1, particle_label.shape[1]))
                    for j in range(0, k_values[k]):
                        min_error = np.amin(errors_copy)
                        best_match = np.argmin(errors_copy)
                        data_label += (particle_label[best_match])  # / min_error)
                        errors_copy[best_match] = 10000
                    pred_test[i][k] = np.argmax(data_label)

            from sklearn.metrics import balanced_accuracy_score  # , accuracy_score, f1_score

            for k in range(len(k_values)):
                train_acc = balanced_accuracy_score(y_train, pred_train[:, k])
                test_acc = balanced_accuracy_score(y_test, pred_test[:, k])
                print_output("{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format
                         (lamb, resol, alpha, seed, fold, epoch, particles.shape[0], total_error / X_train.shape[0],
                          k_values[k], train_acc, test_acc, expname, str(datetime.now()), end_time - start_time))
    fold += 1
f.close()
