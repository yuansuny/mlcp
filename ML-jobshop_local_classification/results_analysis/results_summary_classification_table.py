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

    if obj == "cmax" or obj == "tmax":
        instances = [6, 8, 10, 12, 14]
        sizeStr = ["6x6", "8x8", "10x10", "12x12", "14x14"]
    else:
        instances = [6, 7, 8, 9, 10]
        sizeStr = ["6x6", "7x7", "8x8", "9x9", "10x10"]

    # methods = ["NONE_global_regression", "GPc_pure_classification", "GPr_global_regression",
    #            "MLP_100_pure_classification", "MLP_global_regression", "SVM_pure_classification",
    #            "SVM_global_regression"]
    # methodStr = ["Default", "ClasGP", "RegGP", "MLP_class", "MLP_regr", "SVM_clas", "SVM_regr"]


    methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
               "GP_pure_classification", "GPc_pure_classification", "SVM_pure_classification", "MLP_100_pure_classification",
               "GPr_global_regression", "SVM_global_regression", "MLP_global_regression", "Optimal_pure_regression"]
    methodStr = ["Default", "MinDom", "LowMin", "BranGP", "GP-C", "SVM-C", "MLP-C", "GP-R", "SVM-R", "MLP-R", "OptSol"]


    # methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
    #            "GP_global_regression", "GPc_local_classification", "SVM_local_classification",
    #            "MLP_100_local_classification", "Optimal_global_regression"]
    # methodStr = ["Default", "MinDom", "LowMin", "BranGP-H", "ClasGP-H", "SVM-H", "MLP-H", "OptSol-H"]



    objs = []
    branches = []
    runtimes = []
    solved = []


    bran_ave = []
    bran_std = []
    for instance in instances:
        for ml_method in methods:
            if ml_method == "MLP_global_regression" or ml_method == "SVM_global_regression" or ml_method == "GPr_global_regression":
                df_objs = np.genfromtxt(
                    "../../ML-jobshop_local_regression/results/{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, ml_method,
                                                                                                     instance,
                                                                                                     instance),
                    delimiter=",")
                df_branches = np.genfromtxt(
                    "../../ML-jobshop_local_regression/results/{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h,
                                                                                                         ml_method,
                                                                                                         instance,
                                                                                                         instance),
                    delimiter=",")
            else:
                df_objs = np.genfromtxt(
                    "../results/{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, ml_method, instance, instance),
                    delimiter=",")
                df_branches = np.genfromtxt(
                    "../results/{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, ml_method, instance, instance),
                    delimiter=",")

            objs.append(df_objs)
            branches.append(df_branches)

            bran_ave.append(df_branches.mean())
            bran_std.append(df_branches.std())


    objs = np.array(objs).reshape(len(instances)*len(methods),100)
    branches = np.array(branches).reshape(len(instances)*len(methods),100)

    bran_ave = np.array(bran_ave).reshape(len(instances),  len(methods))
    bran_std = np.array(bran_std).reshape(len(instances), len(methods))

    # print(bran_ave)
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

        val1 = branches[i*len(methods)+idx]
        for j in range(0, len(methods)):
            val2 = branches[i*len(methods)+j]
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

    table = np.array(table).reshape(3*len(instances)+1, len(methods)+3)

    print(table)

    # np.savetxt("results_summary/hybrid_{}_table.csv".format(obj), table, delimiter=",", fmt='%s')
    np.savetxt("results_summary/ml_{}_table.csv".format(obj), table, delimiter=",", fmt='%s')