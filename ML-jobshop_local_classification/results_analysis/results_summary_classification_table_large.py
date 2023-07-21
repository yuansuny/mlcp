from __future__ import print_function
import collections
import numpy as np
import random
import argparse
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='testing.')
    parser.add_argument('-nmachine', default=10, type=int, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-idx', default=0, type=int, help='the index of dataset')
    parser.add_argument('-obj', default="cmax", type=str, help='objective function')  # "cmax", "twt" "tmax"
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance')  # 1.3, 1.5, 1.6
    parser.add_argument('-ml_method', default="DT", type=str,
                        help='ML method')  # "SVM", "DT", "KNN", "MLP", "LR", "NONE"
    parser.add_argument('-level', default="local", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-time_limit', default=60, type=float, help='cutoff time of CP')
    parser.add_argument('-seed', default=10, type=int, help='random seed')
    parser.add_argument('-data_type', default="classification", type=str, help='raw or combine')

    args = parser.parse_args()
    nmachine = args.nmachine
    njob = args.njob
    idx = args.idx
    obj = args.obj
    h = args.duedate
    ml_method = args.ml_method
    time_limit = args.time_limit
    level = args.level
    data_type = args.data_type


    instances = [20, 40, 60, 80, 100]
    sizeStr = ["20x20", "40x40", "60x60", "80x80", "100x100"]

    # methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
    #            "GP_pure_classification", "GPc_pure_classification", "SVM_pure_classification",  "LR_pure_classification",
    #            "MLP_100_pure_classification", "Optimal_pure_regression"]
    # methodStr = ["Default", "MinDom", "LowMin", "BranGP", "ClasGP", "SVM", "LR", "MLP", "OptSol"]


    methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
               "GP_global_regression", "GPc_local_classification", "SVM_local_classification",  "LR_local_classification",
               "MLP_100_local_classification"]
    methodStr = ["Default", "MinDom", "LowMin", "BranGP-H", "ClasGP-H", "SVM-H", "LR-H", "MLP-H"]


    objs = []
    branches = []
    runtimes = []
    solved = []


    bran_ave = []
    bran_std = []
    for instance in instances:
        for ml_method in methods:
            df_objs = np.genfromtxt("../results/{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, ml_method,  instance, instance),delimiter=",")
            df_branches = np.genfromtxt("../results/{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, ml_method, instance, instance),delimiter=",")

            objs.append(df_objs)
            branches.append(df_branches)

            bran_ave.append(df_objs.mean())
            bran_std.append(df_objs.std())


    objs = np.array(objs).reshape(len(instances)*len(methods),100)
    branches = np.array(branches).reshape(len(instances)*len(methods),100)

    bran_ave = np.array(bran_ave).reshape(len(instances),  len(methods))
    bran_std = np.array(bran_std).reshape(len(instances), len(methods))


    header = []
    header.append("Obj")
    header.append("Size")
    header.append("Stats")
    for meth in methodStr:
        header.append(meth)

    table = []
    table.append(header)

    for i in range(0, len(instances)):
        idx = 0
        min_val = bran_ave[i,0]
        for j in range(1, len(methods)-1):
            if bran_ave[i,j] < min_val:
                min_val = bran_ave[i,j]
                idx = j

        ave = []
        std = []
        pvalues = []

        if i == 0:
            ave.append("\multirow{" + "15}" + "{*}" + "{" + obj + "}")
        else:
            ave.append("spaceX")

        ave.append("\multirow{" + "3}" + "{*}" + "{$" + "{}".format(instances[i]) + "\\times" + "{}".format(instances[i]) + "$}")

        ave.append("mean")

        std.append("spaceX")
        std.append("spaceX")
        std.append("std")

        pvalues.append("spaceX")
        pvalues.append("spaceX")
        pvalues.append("p-value")


        val1 = objs[i*len(methods)+idx]
        for j in range(0, len(methods)):
            val2 = objs[i*len(methods)+j]
            (statistic, pvalue) = stats.ttest_rel(val1, val2)
            if pvalue >= 0.05:
                pvalues.append("{:.2e}".format(pvalue))
            else:
                pvalues.append("\\emph{" + "{:.2e}".format(pvalue) + "}")
            if j == idx:
                ave.append(str('\\bf{') + "{:.2e}".format(bran_ave[i, j]) + "}")
            else:
                ave.append("{:.2e}".format(bran_ave[i, j]))
            std.append("{:.2e}".format(bran_std[i, j]))

        table.append(ave)
        table.append(std)
        table.append(pvalues)

    table = np.array(table).reshape(3*len(instances)+1, len(methods))

    print(table)

    np.savetxt("results_summary/hybrid_{}_table_large.csv".format(obj), table, delimiter=",", fmt='%s')