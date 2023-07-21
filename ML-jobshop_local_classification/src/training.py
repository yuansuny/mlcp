import numpy as np
import random
import csv
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import joblib
import argparse
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training ML model.')
    parser.add_argument('-nmachine', default=10, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-obj', default="tmax", type=str, help='objective function')        # "cmax", "twt" "tmax"
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance')     # 1.3, 1.5, 1.6
    parser.add_argument('-ml_method', default="DT", type=str, help='ML method')             # "SVM", "DT", "KNN", "MLP", "LR"
    parser.add_argument('-level', default="local", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-depth', default=7, type=int, help='the maximum depth of DT')
    parser.add_argument('-penalty', default=1.0, type=float, help='regularization parameter of SVM')
    parser.add_argument('-n_neighbors', default=20, type=int, help='number of nearest neighbors for KNN')
    parser.add_argument('-width', default=16, type=int, help='the width of Multi-Layer Perceptron')
    parser.add_argument('-layers', default=8, type=int, help='the number of layers of Multi-Layer Perceptron')
    parser.add_argument('-data_type', default="combine", type=str, help='raw or combine')

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

    if level == "global":
        if data_type == "combine":
            Features = ["opid_diff", "duration_diff", "upstreamdur_diff", "downstreamdur_diff", "totaljobdur_diff",
                          "duedate_diff", "weight_diff", "release_diff", "workload_diff", "maxload_diff"]
            label = []
            opid_diff = []
            duration_diff = []
            upstreamdur_diff = []
            downstreamdur_diff = []
            totaljobdur_diff = []
            duedate_diff = []
            weight_diff = []
            release_diff = []
            workload_diff = []
            maxload_diff = []
            for h in [1.3, 1.5, 1.6]:
                filename = "../training_data/global_datajss_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, nmachine,njob, h)
                if os.path.isfile(filename):
                    with open(filename, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            opid_diff.append(float(row['opid_diff']))
                            duration_diff.append(float(row['duration_diff']))
                            upstreamdur_diff.append(float(row['upstreamdur_diff']))
                            downstreamdur_diff.append(float(row['downstreamdur_diff']))
                            totaljobdur_diff.append(float(row['totaljobdur_diff']))
                            duedate_diff.append(float(row['duedate_diff']))
                            weight_diff.append(float(row['weight_diff']))
                            release_diff.append(float(row['release_diff']))
                            workload_diff.append(float(row['workload_diff']))
                            maxload_diff.append(float(row['maxload_diff']))
                            label.append(float(row['label']))
            training_data = np.reshape(
                opid_diff + duration_diff + upstreamdur_diff + downstreamdur_diff + totaljobdur_diff +
                duedate_diff + weight_diff + release_diff + workload_diff + maxload_diff, [len(Features), len(label)]).T
        elif data_type == "raw":
            Features = ["opid1", "opid2", "duration1", "duration2", "upstreamdur1", "upstreamdur2", "downstreamdur1",
                    "downstreamdur2", "totaljobdur1", "totaljobdur2", "duedate1", "duedate2", "weight1", "weight2",
                    "release1", "release2", "workload1", "workload2", "maxload1", "maxload2"]

            label = []
            opid1 = []
            duration1 = []
            upstreamdur1 = []
            downstreamdur1 = []
            totaljobdur1 = []
            duedate1 = []
            weight1 = []
            release1 = []
            workload1 = []
            maxload1 = []

            opid2 = []
            duration2 = []
            upstreamdur2 = []
            downstreamdur2 = []
            totaljobdur2 = []
            duedate2 = []
            weight2 = []
            release2 = []
            workload2 = []
            maxload2 = []
            for h in [1.3, 1.5, 1.6]:
                filename = "../training_data/global_datajss_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, nmachine,njob, h)
                if os.path.isfile(filename):
                    with open(filename, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            opid1.append(float(row['opid1']))
                            duration1.append(float(row['duration1']))
                            upstreamdur1.append(float(row['upstreamdur1']))
                            downstreamdur1.append(float(row['downstreamdur1']))
                            totaljobdur1.append(float(row['totaljobdur1']))
                            duedate1.append(float(row['duedate1']))
                            weight1.append(float(row['weight1']))
                            release1.append(float(row['release1']))
                            workload1.append(float(row['workload1']))
                            maxload1.append(float(row['maxload1']))
                            opid2.append(float(row['opid2']))
                            duration2.append(float(row['duration2']))
                            upstreamdur2.append(float(row['upstreamdur2']))
                            downstreamdur2.append(float(row['downstreamdur2']))
                            totaljobdur2.append(float(row['totaljobdur2']))
                            duedate2.append(float(row['duedate2']))
                            weight2.append(float(row['weight2']))
                            release2.append(float(row['release2']))
                            workload2.append(float(row['workload2']))
                            maxload2.append(float(row['maxload2']))
                            label.append(float(row['label']))
            training_data = np.reshape(opid1 + opid2 + duration1 + duration2 + upstreamdur1 + upstreamdur2 +
                                       downstreamdur1 + downstreamdur2 + totaljobdur1 + totaljobdur2 + duedate1 +
                                       duedate2 + weight1 + weight2 + release1 + release2 + workload1 + workload2+
                                       maxload1 + maxload2, [len(Features), len(label)]).T

    if level == "local":
        if data_type == "combine":
            Features = ['opid_diff', 'duration_diff', 'upstreamdur_diff', 'downstreamdur_diff', 'totaljobdur_diff',
                        'duedate_diff', 'weight_diff', "release_diff"]
            label = []
            opid_diff = []
            duration_diff = []
            upstreamdur_diff = []
            downstreamdur_diff = []
            totaljobdur_diff = []
            duedate_diff = []
            weight_diff = []
            release_diff = []
        elif data_type == "raw":
            Features = ["opid1", "opid2", "duration1", "duration2", "upstreamdur1", "upstreamdur2", "downstreamdur1",
                        "downstreamdur2", "totaljobdur1", "totaljobdur2", "duedate1", "duedate2", "weight1", "weight2",
                        "release1", "release2"]
            label = []
            opid1 = []
            duration1 = []
            upstreamdur1 = []
            downstreamdur1 = []
            totaljobdur1 = []
            duedate1 = []
            weight1 = []
            release1 = []
            opid2 = []
            duration2 = []
            upstreamdur2 = []
            downstreamdur2 = []
            totaljobdur2 = []
            duedate2 = []
            weight2 = []
            release2 = []
        for h in [1.3, 1.5, 1.6]:
            if level == "global":
                filename = "../training_data/global_datajss_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, nmachine, njob, h)
            elif level == "local":
                filename = "../training_data/datajss_{}_{}_nm_{}_nj_{}_h{}.csv".format(obj, data_type, nmachine, njob, h)
                if os.path.isfile(filename):
                    if data_type == "combine":
                        with open(filename, newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                opid_diff.append(float(row['opid_diff']))
                                duration_diff.append(float(row['duration_diff']))
                                upstreamdur_diff.append(float(row['upstreamdur_diff']))
                                downstreamdur_diff.append(float(row['downstreamdur_diff']))
                                totaljobdur_diff.append(float(row['totaljobdur_diff']))
                                duedate_diff.append(float(row['duedate_diff']))
                                weight_diff.append(float(row['weight_diff']))
                                release_diff.append(float(row['release_diff']))
                                label.append(float(row['label']))
                        training_data = np.reshape(
                            opid_diff + duration_diff + upstreamdur_diff + downstreamdur_diff + totaljobdur_diff +
                            duedate_diff + weight_diff + release_diff, [len(Features), len(label)]).T
                    elif data_type == "raw":
                        with open(filename, newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                opid1.append(float(row['opid1']))
                                duration1.append(float(row['duration1']))
                                upstreamdur1.append(float(row['upstreamdur1']))
                                downstreamdur1.append(float(row['downstreamdur1']))
                                totaljobdur1.append(float(row['totaljobdur1']))
                                duedate1.append(float(row['duedate1']))
                                weight1.append(float(row['weight1']))
                                release1.append(float(row['release1']))
                                opid2.append(float(row['opid2']))
                                duration2.append(float(row['duration2']))
                                upstreamdur2.append(float(row['upstreamdur2']))
                                downstreamdur2.append(float(row['downstreamdur2']))
                                totaljobdur2.append(float(row['totaljobdur2']))
                                duedate2.append(float(row['duedate2']))
                                weight2.append(float(row['weight2']))
                                release2.append(float(row['release2']))
                                label.append(float(row['label']))
        training_data = np.reshape(opid1 + opid2 + duration1 + duration2 + upstreamdur1 + upstreamdur2 +
               downstreamdur1 + downstreamdur2 + totaljobdur1 + totaljobdur2 + duedate1 +
               duedate2 + weight1 + weight2 + release1 + release2, [len(Features), len(label)]).T

    label = np.array(label)
    print("number of training instance is ", len(label))

    if ml_method == "SVM":
        print("linear SVM; penalty is : ", penalty)
        # ml_model = svm.SVC(kernel='linear', C=penalty, dual=True)
        ml_model = LinearSVC(dual=False, tol=0.0001, C=penalty, max_iter=100000)
    elif ml_method == "DT":
        print("DT; depth: ", depth)
        ml_model = tree.DecisionTreeClassifier(max_depth=depth)
    elif ml_method == "KNN":
        print("KNN; k: ", n_neighbors)
        ml_model = neighbors.KNeighborsClassifier(n_neighbors)
    elif ml_method == "MLP":
        print("MLP; size: ", width, layers)
        ml_model = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(width, layers), max_iter=100000)
    elif ml_method == "LR":
        print("LR; penalty is : ", penalty)
        ml_model = LogisticRegression(C=penalty,max_iter=100000)

    ml_model.fit(training_data, label)
    joblib.dump(ml_model, "../training_data/{}_{}_{}_{}.pkl".format(obj, ml_method, level, data_type))
    # # 10-fold cross validation accuracy
    k = random.randint(0, 10000)
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=k)
    scores = cross_val_score(ml_model, training_data, label, cv=cv)
    print ("10-fold cross validation accuracy of {} on {} is : {}".format(ml_method, obj, scores.mean()))