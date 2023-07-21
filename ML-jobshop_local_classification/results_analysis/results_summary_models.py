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


    # methods = ["NONE_global_regression", "GP_pure_classification", "GPc_pure_classification", "GPr_global_regression",
    #            "MLP_100_pure_classification", "MLP_global_regression", "SVM_pure_classification", "SVM_global_regression"]
    # methodStr = ["Default", "BranGP_pure", "ClasGP", "RegGP", "MLP_class", "MLP_regr", "SVM_clas", "SVM_regr"]

    # methods = ["NONE_global_regression", "GPc_pure_classification", "GPr_global_regression",
    #            "MLP_100_pure_classification", "MLP_global_regression", "SVM_pure_classification", "SVM_global_regression"]
    # methodStr = ["Default", "ClasGP", "RegGP", "MLP_class", "MLP_regr", "SVM_clas", "SVM_regr"]


    methods = ["NONE_global_regression", "GPc_pure_classification", "GPr_global_regression",
               "MLP_100_pure_classification", "MLP_global_regression", "SVM_pure_classification", "SVM_global_regression"]
    methodStr = ["Default", "GP-C", "GP-R", "MLP-C", "MLP-R", "SVM-C", "SVM-R"]


    objs = []
    branches = []
    runtimes = []
    solved = []


    for instance in instances:
        for ml_method in methods:
            if ml_method == "MLP_global_regression" or ml_method == "SVM_global_regression" or ml_method == "GPr_global_regression":
                df_objs = np.genfromtxt(
                    "../../ML-jobshop_local_regression/results/{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, ml_method, instance, instance),
                    delimiter=",")
                df_branches = np.genfromtxt(
                    "../../ML-jobshop_local_regression/results/{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, ml_method, instance, instance),
                    delimiter=",")
            else:
                df_objs = np.genfromtxt("../results/{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, ml_method,  instance, instance),delimiter=",")
                df_branches = np.genfromtxt("../results/{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, ml_method, instance, instance),delimiter=",")

            objs.append(df_objs)
            branches.append(df_branches)


    objs = np.array(objs).reshape(len(instances)*len(methods),100)
    branches = np.array(branches).reshape(len(instances)*len(methods),100)

    # print(objs)

    # ax = sns.boxplot(x=branches[1,:])

    # print(branches)

    imp = []
    for i in range(len(instances)):
        for j in range(0,len(methods)):
            for k in range(100):
                id1 = i*len(methods)+j
                id2 = i*len(methods)
                # imp.append(branches[id1, k])
                imp.append((branches[id2,k] - branches[id1,k])/max(branches[id2,k],branches[id1,k]))
                # imp.append((objs[id2, k] - objs[id1, k]) / max(objs[id2, k], objs[id1, k]))
                # imp.append((branches[id2, k] - branches[id1, k]) / branches[id2, k])

    # imp = np.array(imp).reshape(len(instances) * (len(methods)-1), 100)
    imp = np.array(imp).reshape(len(instances) * (len(methods)), 100)
    # imp = np.transpose(imp)

    for i in range(len(instances) * len(methods)):
        print(imp[i,:].mean())


    size = []
    for i in range(len(instances)):
        for j in range(1, len(methods)):
            size.append(sizeStr[i])
    print(size)



    method = []
    for i in range(len(instances)):
        for j in range(1, len(methods)):
            method.append(methodStr[j])
    print(method)

    improvement = []
    for i in range(len(instances)):
        for j in range(1, len(methods)):
            improvement.append(imp[i*len(methods)+j])

    # plt.plot()
    # # ax = sns.boxplot(order=["6x6", "8x8", "10x10"], data=df)
    # ax = sns.boxplot(data=improvement)
    # plt.show()


    data = {'Instance Size':size,
            'Method':method,
            'Improvement':improvement
            }


    # df = pd.DataFrame(data['Improvement'])
    df = pd.DataFrame(data)


    df = df.explode('Improvement')
    df['Improvement'] = df['Improvement'].astype('float')

    # print(df)

    # plt.plot()
    plt.figure(figsize=(8, 8))

    # ax = sns.boxplot(order=["6x6", "8x8", "10x10"], data=df)
    ax = sns.boxplot(x='Instance Size', y='Improvement', hue='Method', data=df)
    # ax = sns.violinplot(x='Size', y='Improvement', hue='Method', data=df)

    ax.legend(loc='lower left', fontsize=19, ncol=3)
    plt.xlabel('Instance Size', fontsize=20)
    plt.ylabel('Improvement', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    filename = 'figs/MLmodels_' + obj + '_imp.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


    table = []
    ave = []
    std = []
    pvalues = []

    val1 = df.Improvement[df.Method == methodStr[3]]
    for j in range(1, len(methods)):
        val2 = df.Improvement[df.Method == methodStr[j]]
        (statistic, pvalue) = stats.ttest_rel(val1, val2)
        ave.append(val2.mean())
        std.append(val2.std())
        pvalues.append(pvalue)
        print(val1.mean(), val2.mean(), pvalue)

    table.append(methodStr[1:len(methods)])
    table.append(ave)
    table.append(std)
    table.append(pvalues)

    table = np.array(table).reshape(4, len(methods)-1)

    print(table)
    # np.savetxt("results_summary/optimal_{}_table.csv".format(obj), table, delimiter=",", fmt='%s')






    # np.savetxt("cmax_optimal_{}_ave_objs.csv".format(level), np.transpose(objs), delimiter=",")
    # np.savetxt("cmax_optimal_{}_ave_branches.csv".format(level), np.transpose(branches), delimiter=",")
    # np.savetxt("cmax_optimal_{}_ave_runtimes.csv".format(level), np.transpose(runtimes), delimiter=",")

    #
    # print('==================================')
    # print("Average running time --- %s seconds ---" % alltime.mean())
    # print("Average number of branches --- %s ---" % allbranches.mean())
    # print("Average objective values --- %s ---" % allobjs.mean())

