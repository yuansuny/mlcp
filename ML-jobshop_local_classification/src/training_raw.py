import numpy as np
import random
import csv
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
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
    parser.add_argument('-ml_method', default="DT", type=str, help='ML method')             # "SVM", "DT", "KNN", "MLP"
    parser.add_argument('-level', default="global", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-depth', default=7, type=int, help='the maximum depth of DT')
    parser.add_argument('-penalty', default=1, type=int, help='regularization parameter of SVM')
    parser.add_argument('-n_neighbors', default=20, type=int, help='number of nearest neighbors for KNN')
    parser.add_argument('-width', default=16, type=int, help='the width of Multi-Layer Perceptron')
    parser.add_argument('-layers', default=8, type=int, help='the number of layers of Multi-Layer Perceptron')

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

    Features = ["opid1", "opid2", "duration1", "duration2", "upstreamdur1", "upstreamdur2", "downstreamdur1", "downstreamdur2", "totaljobdur1", "totaljobdur2", "duedate1", "duedate2", "weight1", "weight2", "release1", "release2"]
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
            filename = "../training_data/global_datajss_{}_raw_nm_{}_nj_{}_h{}.csv".format(obj, nmachine, njob, h)
        else:
            filename = "../training_data/datajss_{}_raw_nm_{}_nj_{}_h{}.csv".format(obj, nmachine, njob, h)
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

                    opid2.append(float(row['opid2']))
                    duration2.append(float(row['duration2']))
                    upstreamdur2.append(float(row['upstreamdur2']))
                    downstreamdur2.append(float(row['downstreamdur2']))
                    totaljobdur2.append(float(row['totaljobdur2']))
                    duedate2.append(float(row['duedate2']))
                    weight2.append(float(row['weight2']))

                    label.append(float(row['label']))

    training_data = np.reshape(opid1 + opid2 + duration1 + duration2 + upstreamdur1 + upstreamdur2 +
                               downstreamdur1 + downstreamdur2 + totaljobdur1 + totaljobdur2 +
                               duedate1 + duedate2 + weight1 + weight2, [len(Features), len(label)]).T
    label = np.array(label)
    print("number of training instance is ", len(label))

    if ml_method == "SVM":
        print("linear SVM; penalty is : ", penalty)
        ml_model = svm.SVC(kernel='linear', C=penalty)
    elif ml_method == "DT":
        print("DT; depth: ", depth)
        ml_model = tree.DecisionTreeClassifier(max_depth=depth)
    elif ml_method == "KNN":
        print("KNN; k: ", n_neighbors)
        ml_model = neighbors.KNeighborsClassifier(n_neighbors)
    elif ml_method == "MLP":
        print("MLP; size: ", width, layers)
        ml_model = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(width, layers), max_iter=100000)

    ml_model.fit(training_data, label)
    joblib.dump(ml_model, "../training_data/{}_{}_{}_raw.pkl".format(obj, ml_method, level))
    # # 10-fold cross validation accuracy
    k = random.randint(0, 10000)
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=k)
    scores = cross_val_score(ml_model, training_data, label, cv=cv)
    print ("10-fold cross validation accuracy of {} on {} is : {}".format(ml_method, obj, scores.mean()))