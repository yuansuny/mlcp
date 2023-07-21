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

from gplearn import genetic

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
    parser.add_argument('-level', default="global", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-depth', default=7, type=int, help='the maximum depth of DT')
    parser.add_argument('-penalty', default=1.0, type=float, help='regularization parameter of SVM')
    parser.add_argument('-n_neighbors', default=20, type=int, help='number of nearest neighbors for KNN')
    parser.add_argument('-width', default=16, type=int, help='the width of Multi-Layer Perceptron')
    parser.add_argument('-layers', default=100, type=int, help='the number of layers of Multi-Layer Perceptron')
    parser.add_argument('-data_type', default="regression", type=str, help='raw or combine')

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
            filename = "../training_data_nor/training_data_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, n, n, h)
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
        duedate + weight + release + earliest_start + workload, [len(Features), len(label)]).T


    label = np.array(label)
    print("number of training instance is ", len(label))

    if ml_method == "SVM":
        print("SVM; penalty is : ", penalty)
        # ml_model = svm.SVC(kernel='linear', C=penalty, dual=True)
        # ml_model = LinearSVC(dual=False, tol=0.0001, C=penalty, max_iter=100000)
        # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default =’rbf’
        # ml_model = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=penalty, epsilon=0.1)
        ml_model = svm.LinearSVR(C=1.0, max_iter=100000)
    elif ml_method == "OLS":
        print("OLS : ")
        ml_model = linear_model.LinearRegression()
        # ml_model = linear_model.Ridge(alpha=.5)
    elif ml_method == "SGD":
        print("SGD " )
        ml_model = linear_model.SGDRegressor(loss='squared_loss', penalty='l2')
    elif ml_method == "KNN":
        print("KNN; k: ", n_neighbors)
        ml_model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
    elif ml_method == "DT":
        print("DT; depth: ", depth)
        ml_model = tree.DecisionTreeRegressor(max_depth=depth)
    elif ml_method == "GPr":
        print("GPr")
        ml_model = genetic.SymbolicRegressor(low_memory=True, n_jobs=4, generations=2000)
    elif ml_method == "MLP":
        print("MLP; size: ", layers)
        # ml_model = neural_network.MLPRegressor(hidden_layer_sizes=(64, 32, 16, 8), max_iter=10000)
        ml_model = neural_network.MLPRegressor(hidden_layer_sizes=(layers,), max_iter=1000000)
        # ml_model = neural_network.MLPRegressor(hidden_layer_sizes=(16, 16, 16, 16), max_iter=10000)
        # ml_model = neural_network.MLPRegressor(hidden_layer_sizes=(16, 16, 16, 16), max_iter=10000)
        # ml_model = neural_network.MLPRegressor(hidden_layer_sizes=(layers, layers, layers, layers, layers, layers), max_iter=100000)

    # elif ml_method == "DT":
    #     print("DT; depth: ", depth)
    #     ml_model = tree.DecisionTreeClassifier(max_depth=depth)
    # elif ml_method == "KNN":
    #     print("KNN; k: ", n_neighbors)
    #     ml_model = neighbors.KNeighborsClassifier(n_neighbors)
    # elif ml_method == "MLP":
    #     print("MLP; size: ", width, layers)
    #     ml_model = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(width, layers), max_iter=100000)
    # elif ml_method == "LR":
    #     print("LR; penalty is : ", penalty)
    #     ml_model = LogisticRegression(C=penalty,max_iter=100000)

    ml_model.fit(training_data, label)

    # scores = cross_val_score(ml_model, training_data, label, cv=5, scoring= 'r2')
    # print(scores.mean())

    # joblib.dump(ml_model, "../training_data/{}_{}_{}_{}.pkl".format(obj, ml_method, level, data_type))

    # # # 10-fold cross validation accuracy
    # k = random.randint(0, 10000)
    # cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=k)
    # scores = cross_val_score(ml_model, training_data, label, cv=cv)
    # print ("10-fold cross validation accuracy of {} on {} is : {}".format(ml_method, obj, scores.mean()))