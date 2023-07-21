from __future__ import print_function
import collections
import numpy as np
import random
import argparse
import pandas as pd
from scipy.stats import ttest_ind

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='testing.')
    parser.add_argument('-nmachine', default=10, type=int, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-obj', default="cmax", type=str, help='objective function')  # "cmax", "twt" "tmax"
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance')  # 1.3, 1.5, 1.6
    parser.add_argument('-level', default="global", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-scale', default="small", type=str, help='data scale') # "small", "large"

    args = parser.parse_args()
    nmachine = args.nmachine
    njob = args.njob
    obj = args.obj
    h = args.duedate
    scale = args.scale
    level = args.level

    h = 1.4
    if scale == "small":
        instances = range(5, 15)
        cutoff = 3600
        if obj == "cmax":
            ml_methods = ["DT_global", "DT_local", "SVM_local", "SVM_global_regression", "OLS_global_regression",
                          "KNN_global_regression", "DT_global_regression", "MLP_global_regression",
                          "MLP_100_global_regression", "MLP_512_global_regression", "MLP_1024_global_regression",
                          "MLP_2048_global_regression", "GP_global"]
            h = 1.3
        else:
            ml_methods = ["NONE_global", "DT_global", "DT_local", "SVM_local", "GP_global"]
    else:
        instances = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        ml_methods = ["NONE_global", "DT_local", "SVM_local", "SVM_global_regression", "OLS_global_regression",
                      "KNN_global_regression", "DT_global_regression", "MLP_global_regression",
                      "MLP_100_global_regression", "MLP_512_global_regression", "MLP_1024_global_regression",
                      "MLP_2048_global_regression", "GP_global"]
        cutoff = 60

    objs = []
    branches = []
    runtimes = []
    solved = []

    for method in ml_methods:
        for instance in instances:
            if obj == 'cmax' and method != 'DT_global' and method != 'DT_local':
                h = 1.4
            df_objs = np.genfromtxt("../results/{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, method, instance, instance),delimiter=",")
            df_branches = np.genfromtxt("../results/{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, method, instance, instance),delimiter=",")
            df_runtimes = np.genfromtxt("../results/{}_{}_{}_nm_{}_nj_{}_runtimes.csv".format(obj, h, method, instance, instance),delimiter=",")

            objs.append(df_objs.mean())
            branches.append(df_branches.mean())
            runtimes.append(df_runtimes.mean())
            solved.append(sum(time < cutoff for time in df_runtimes))

    objs = np.array(objs).reshape(len(ml_methods),len(instances))
    branches = np.array(branches).reshape(len(ml_methods), len(instances))
    runtimes = np.array(runtimes).reshape(len(ml_methods), len(instances))
    solved = np.array(solved).reshape(len(ml_methods), len(instances))

    print(objs)
    print(branches)
    print(runtimes)
    print(solved)

    # print(ttest_ind(branches[len(ml_methods)-1,:], branches[len(ml_methods)-3,:]))

    np.savetxt("../results/{}_{}_ave_objs.csv".format(obj, scale), objs, delimiter=",")
    np.savetxt("../results/{}_{}_ave_branches.csv".format(obj, scale), branches,delimiter=",")
    np.savetxt("../results/{}_{}_ave_runtimes.csv".format(obj, scale), runtimes,delimiter=",")
    np.savetxt("../results/{}_{}_num_solved.csv".format(obj, scale),  solved, delimiter=",")


