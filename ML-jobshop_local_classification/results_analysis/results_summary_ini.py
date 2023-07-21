from __future__ import print_function
import collections
import numpy as np
import random
import argparse
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

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

    h = 1.3

    # ml_methods = ["NONE_global_regression", "GP_global_regression", "SVM_global_regression", "OLS_global_regression",
    #               "DT_global_regression", "KNN_global_regression", "MLP_128_global_regression", "MLP_256_global_regression",
    #               "MLP_512_global_regression", "MLP_1024_global_regression", "MLP_2048_global_regression"]

    # ml_methods = ["NONE_local_classification", "GP_local_classification", "LR_local_classification", "MLP_local_classification"]
    ml_methods = ["NONE_local_classification", "GP_local_classification", "GPc_local_classification", "LR_local_classification"]

    # ml_methods = ["NONE_global_regression", "GP_global_regression",
    #               "DT_global_regression", "KNN_global_regression", "MLP_512_global_regression"]

    # ml_methods = ["NONE_global_regression", "MLP_128_global_regression", "MLP_256_global_regression",
    #               "MLP_512_global_regression", "MLP_1024_global_regression", "MLP_2048_global_regression"]

    # ml_methods = ["NONE_global_regression", "GP_global_regression", "SVM_global_regression", "OLS_global_regression",
    #               "DT_global_regression", "KNN_global_regression"]

    if scale == "small":
        # instances = range(5, 15)
        instances = [5,6,7,8,9,10,11,12,13,14,20,30,40,50]
        # instances = range(5,11)
        # instances = range(11, 21)
        # instances = range(5, 14)
        # instances = range(5, 13)
        cutoff = 3600
    else:
        instances = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        # instances = [40, 50, 60, 70, 80, 90, 100]
        cutoff = 60

    objs = []
    branches = []
    runtimes = []
    solved = []

    for method in ml_methods:
        for instance in instances:
            df_objs = np.genfromtxt("../results_ini/{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, method, instance, instance),delimiter=",")
            df_branches = np.genfromtxt("../results_ini/{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, method, instance, instance),delimiter=",")
            df_runtimes = np.genfromtxt("../results_ini/{}_{}_{}_nm_{}_nj_{}_runtimes.csv".format(obj, h, method, instance, instance),delimiter=",")

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

    for i in range(len(ml_methods)):
        print(sum(solved[i,:]))

    # improvement ratio
    imp = []
    for i in range(len(ml_methods)):
        for j in range(len(objs[i, :])):
            imp.append((objs[0, j] - objs[i, j]) / objs[0, j])
    # if scale == "small":
    #     for i in range(len(ml_methods)):
    #         for j in range(len(branches[i, :])):
    #             imp.append((branches[0, j] - branches[i, j]) / branches[0, j])
    #     # for i in range(len(ml_methods)):
    #     #     for j in range(len(runtimes[i, :])):
    #     #         imp.append((runtimes[0, j] - runtimes[i, j]) / runtimes[0, j])
    # elif scale == "large":
    #     for i in range(len(ml_methods)):
    #         for j in range(len(objs[i, :])):
    #             imp.append((objs[0, j] - objs[i, j]) / objs[0, j])

    imp = np.array(imp).reshape(len(ml_methods), len(instances))


    # statistical information and average
    pvalues = []
    ave = []

    for i in range(len(ml_methods)):
        ave.append(imp[i, :].mean())
        (statistic, pvalue) = stats.ttest_rel(imp[i, :],imp[0, :])
        pvalues.append(pvalue)
        (statistic, pvalue) = stats.ttest_rel(imp[i, :], imp[1, :])
        pvalues.append(pvalue)

    # if scale == "small":
    #     for i in range(len(ml_methods)):
    #         ave.append(branches[i,:].mean())
    #         (statistic, pvalue) = stats.ttest_rel(branches[i,:], branches[0,:])
    #         pvalues.append(pvalue)
    #         (statistic, pvalue) = stats.ttest_rel(branches[i, :], branches[1, :])
    #         pvalues.append(pvalue)
    # elif scale == "large":
    #     for i in range(len(ml_methods)):
    #         ave.append(objs[i, :].mean())
    #         (statistic, pvalue) = stats.ttest_rel(objs[i, :], objs[0, :])
    #         pvalues.append(pvalue)
    #         (statistic, pvalue) = stats.ttest_rel(objs[i, :], objs[1, :])
    #         pvalues.append(pvalue)

    ave = np.array(ave).reshape(len(ml_methods), 1)
    pvalues = np.array(pvalues).reshape(len(ml_methods), 2)

    print(ave)
    print(pvalues)


    # create table for the paper
    table = []

    for i in range(len(ml_methods)):
        for v in imp[i, :]:
            table.append(v)
        table.append(ave[i, 0])
        table.append(pvalues[i, 0])
        table.append(pvalues[i, 1])

    table = np.array(table).reshape(len(ml_methods), len(instances)+3)

    print(table)



    np.savetxt("results_summary/{}_{}_ave_objs.csv".format(obj, scale), objs, delimiter=",")
    np.savetxt("results_summary/{}_{}_ave_branches.csv".format(obj, scale), branches,delimiter=",")
    np.savetxt("results_summary/{}_{}_ave_runtimes.csv".format(obj, scale), runtimes,delimiter=",")
    np.savetxt("results_summary/{}_{}_num_solved.csv".format(obj, scale),  solved, delimiter=",")
    np.savetxt("results_summary/{}_{}_table.csv".format(obj, scale), table, delimiter=",")









    # plot branches
    plt.figure(figsize=(16, 16))

    target_names = ["NONE", "GP", "GPc", "LR"]

    # target_names = ["NONE", "GP", "SVM", "OLS", "DT", "KNN", "MLP_128", "MLP_256", "MLP_512", "MLP_1024", "MLP_2048"]

    # target_names = ["NONE", "MLP_128", "MLP_256", "MLP_512", "MLP_1024", "MLP_2048"]

    plot_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:purple', 'black']

    for i, color, target_name in zip(range(len(ml_methods)), plot_colors, target_names):
        if i > 0:
            # plt.plot(instances, branches[i, :], c=color, label=target_name)
            plt.plot(range(len(instances)), imp[i, :]*100, c=color, label=target_name)
            # plt.scatter(X[idx, 0], X[idx, 1], c=color, lw=2, label=target_name, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize=30)
    plt.xticks(range(len(instances)), instances, fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("problem size", fontsize=30)
    plt.ylabel("objective improvement %", fontsize=30)
    filename = 'figs/' + obj + '_' + scale + '_obj.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()



