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

    # ml_methods = ["NONE_local", "GP_local", "SVM_local", "DT_local", "SVM_regression", "MLP_1024_regression","MLP_512_regression",
    #               "MLP_200_regression", "MLP_100_regression", "MLP_50_regression", "OLS_regression", "DT_regression",
    #               "KNN_regression"]
    ml_methods = ["NONE_local", "GP_local", "DT_local", "SVM_local", "MLP_2048_regression", "MLP_1024_regression",
                  "MLP_512_regression", "MLP_200_regression", "MLP_100_regression", "MLP_50_regression"]
    # ml_methods = ["NONE_local", "GP_local", "DT_local", "SVM_local", "MLP_2048_regression", "MLP_1024_regression"]
    cutoff = 60

    objs = []
    branches = []
    runtimes = []
    solved = []

    objs_all = []
    runtimes_all = []

    for method in ml_methods:
        df_objs = np.genfromtxt("../results/benchmark_{}_{}_objs.csv".format(obj, method),delimiter=",")
        df_branches = np.genfromtxt("../results/benchmark_{}_{}_branches.csv".format(obj, method),delimiter=",")
        df_runtimes = np.genfromtxt("../results/benchmark_{}_{}_runtimes.csv".format(obj, method),delimiter=",")

        objs.append(df_objs.mean())
        branches.append(df_branches.mean())
        runtimes.append(df_runtimes.mean())
        solved.append(sum(time < cutoff - 0.1 for time in df_runtimes))

        objs_all.append(df_objs)
        runtimes_all.append(df_runtimes)

    # objs = np.array(objs).reshape(len(ml_methods),len(instances))
    # branches = np.array(branches).reshape(len(ml_methods), len(instances))
    # runtimes = np.array(runtimes).reshape(len(ml_methods), len(instances))
    # solved = np.array(solved).reshape(len(ml_methods), len(instances))

    print(objs)
    print(branches)
    print(runtimes)
    print(solved)

    # print(objs_all)

    np.savetxt("../results/benchmark_{}_all_objs.csv".format(obj), np.transpose(objs_all), delimiter=",")
    np.savetxt("../results/benchmark_{}_all_runtimes.csv".format(obj), np.transpose(runtimes_all), delimiter=",")

    # print(ttest_ind(branches[len(ml_methods)-1,:], branches[len(ml_methods)-3,:]))

    # np.savetxt("../results/{}_{}_ave_objs.csv".format(obj, scale), objs, delimiter=",")
    # np.savetxt("../results/{}_{}_ave_branches.csv".format(obj, scale), branches,delimiter=",")
    # np.savetxt("../results/{}_{}_ave_runtimes.csv".format(obj, scale), runtimes,delimiter=",")
    # np.savetxt("../results/{}_{}_num_solved.csv".format(obj, scale),  solved, delimiter=",")


