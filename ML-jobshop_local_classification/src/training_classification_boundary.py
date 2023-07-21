import numpy as np
import random
import csv
from sklearn import svm
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import neighbors
from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import naive_bayes

from gplearn import genetic

from sklearn.decomposition import PCA, IncrementalPCA

import matplotlib.pyplot as plt

from matplotlib import colors


import joblib
import argparse
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training ML model.')
    parser.add_argument('-nmachine', default=10, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-obj', default="tmax", type=str, help='objective function')        # "cmax", "twt" "tmax"
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance')     # 1.3, 1.5, 1.6
    parser.add_argument('-ml_method', default="SVM", type=str, help='ML method')             # "SVM", "DT", "KNN", "MLP", "LR"
    parser.add_argument('-level', default="local", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-depth', default=7, type=int, help='the maximum depth of DT')
    parser.add_argument('-penalty', default=1.0, type=float, help='regularization parameter of SVM')
    parser.add_argument('-n_neighbors', default=20, type=int, help='number of nearest neighbors for KNN')
    parser.add_argument('-width', default=16, type=int, help='the width of Multi-Layer Perceptron')
    parser.add_argument('-layers', default=100, type=int, help='the number of layers of Multi-Layer Perceptron')
    parser.add_argument('-generations', default=2000, type=int, help='number of generation of GP')
    parser.add_argument('-data_type', default="classification", type=str, help='raw or combine')

    args = parser.parse_args()
    nmachine = args.nmachine
    njob = args.njob
    obj = args.obj
    h = args.duedate
    ml_method = args.ml_method
    depth = args.depth
    penalty = args.penalty
    n_neighbors = args.n_neighbors
    width = args.width
    layers = args.layers
    generations = args.generations
    level = args.level
    data_type = args.data_type

    Features = ["opid", "duration", "upstreamdur", "downstreamdur", "totaljobdur", "duedate", "weight", "release", "earliest_start", "workload"]
    label = []
    opid = []
    duration = []
    upstreamdur = []
    downstreamdur = []
    totaljobdur = []
    duedate = []
    weight = []
    release = []
    earliest_start = []
    workload = []
    # for n in [5, 6, 7, 8, 9, 10]:
    for n in [9]:
        # for h in [1.3, 1.5, 1.6]:
        for h in [1.3]:
            # filename = "../training_data/global_datajss_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, nmachine, njob, h)
            # filename = "../training_data/MLP_global_datajss_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, n, n, h)
            if ml_method == "GPc" or ml_method == "DT":
                filename = "../training_data/training_data_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, n, n, h)
            else:
                filename = "../training_data_norm/training_data_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, n, n, h)

            if os.path.isfile(filename):
                with open(filename, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        opid.append(float(row['opid']))
                        duration.append(float(row['duration']))
                        upstreamdur.append(float(row['upstreamdur']))
                        downstreamdur.append(float(row['downstreamdur']))
                        totaljobdur.append(float(row['totaljobdur']))
                        duedate.append(float(row['duedate']))
                        weight.append(float(row['weight']))
                        release.append(float(row['release']))
                        earliest_start.append(float(row['earliest_start']))
                        workload.append(float(row['workload']))
                        label.append(float(row['label']))
    training_data = np.reshape(
        opid + duration + upstreamdur + downstreamdur + totaljobdur +
        duedate + weight + release + earliest_start, [len(Features)-1, len(label)]).T


    label = np.array(label)
    print("number of training instance is ", len(label))

    n = 10000
    indices = np.random.permutation(np.arange(len(label)))[:n]
    training_data = training_data[indices,:]
    label = label[indices]

    if ml_method == "SVM":
        print("SVM; penalty is : ", penalty)
        # ml_model = svm.SVC(kernel='rbf', C=penalty)
        # 'linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        ml_model = LinearSVC(dual=False, tol=0.0001, C=penalty, max_iter=100000)
    elif ml_method == "KNN":
        print("KNN; k: ", n_neighbors)
        ml_model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    elif ml_method == "DT":
        print("DT; depth: ", depth)
        ml_model = tree.DecisionTreeClassifier(max_depth=depth)
    elif ml_method == "GPc":
        print("GPc; generations: ", generations)
        ml_model = genetic.SymbolicClassifier(low_memory=True, n_jobs=4, generations=generations)
    elif ml_method == "GPcn":
        print("GPcn; generations: ", generations)
        ml_model = genetic.SymbolicClassifier(low_memory=True, n_jobs=4, generations=generations)
    elif ml_method == "RF":
        print("RF; depth: ", depth)
        ml_model = ensemble.RandomForestClassifier(n_estimators=10, max_depth=depth)
    elif ml_method == "MLP":
        print("MLP; size: ", layers)
        ml_model = neural_network.MLPClassifier(hidden_layer_sizes=(layers,), max_iter=1000000)
        # ml_model = neural_network.MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8), early_stopping=True, max_iter=1000)
        # ml_model = neural_network.MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8), max_iter=100000)
        # ml_model = neural_network.MLPClassifier(hidden_layer_sizes=(2048), early_stopping=True, max_iter=1000)
    elif ml_method == "LR":
        print("LR; penalty is : ", penalty)
        ml_model = linear_model.LogisticRegression(C=penalty,max_iter=100000)
    elif ml_method == "NB":
        print("NB;")
        ml_model = naive_bayes.GaussianNB()




    n_components = 2
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(training_data)

    variance_explained = pca.explained_variance_ratio_
    print("Variance Explained by Each Principal Component:")
    for i, explained_var in enumerate(variance_explained):
        print(f"Principal Component {i + 1}: {explained_var:.2f}")

    n_classes = 2
    plot_colors = ['mediumblue', 'maroon']
    target_names = ['Negative', 'Positive']

    plot_step = 0.02
    depth = 4
    X = X_pca[:, [0, 1]]
    y = np.array(label)

    # plt.figure(figsize=(8, 8))
    # for i, color, target_name in zip([1, 0], plot_colors, target_names):
    #     idx = np.where(y == i)
    #     # plt.scatter(X[idx, 0], X[idx, 1], c=color, lw=2, label=target_name, edgecolor='black')
    #     plt.scatter(X[idx, 0], X[idx, 1], c=color, s=1, lw=2, label=target_name)
    #
    # plt.xlabel('$Z_1$', fontsize=12)
    # plt.ylabel('$Z_2$', fontsize=12)
    # plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize=12)
    # plt.savefig("PCAVis", bbox_inches='tight')
    #
    # plt.figure(figsize=(8, 8))
    # np.savetxt("PCA_data.csv", X, delimiter=",")

    ml_model.fit(X, label)

    Zpred = ml_model.predict(X)


    plt.figure(figsize=(8, 8))

    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    cmap = colors.ListedColormap(['lightsteelblue', 'lightcoral'])
    # bounds = [0, 1, 2, 3]

    Z = ml_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # cs = plt.contourf(xx, yy, Z, vmin=0, vmax=1, cmap=plt.cm.RdYlBu)
    # cs = plt.contourf(xx, yy, Z, vmin=0, vmax=1, cmap=plt.cm.RdYlBu)
    cs = plt.contourf(xx, yy, Z, vmin=0, vmax=1, cmap=cmap)

    # plt.xlabel(feature_names[4], fontsize=16)
    # plt.ylabel(feature_names[7], fontsize=16)


    for i, color, target_name in zip([0, 1], plot_colors, target_names):
        idx = np.where(y == i)
        # plt.scatter(X[idx, 0], X[idx, 1], c=color, lw=2, label=target_name, edgecolor='black')
        plt.scatter(X[idx, 0], X[idx, 1], c=color, s=2, lw=2, label=target_name)
        # plt.scatter(X[idx, 0], X[idx, 1], c=color, s=1, lw=2, label=target_name)

    plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize=20, markerscale=5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('$Z_1$', fontsize=20)
    plt.ylabel('$Z_2$', fontsize=20)

    # plt.title(': SVM boundary', fontsize=20)
    filename = '../results_analysis/figs/{}_{}_Boundary.png'.format(obj, ml_method)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()