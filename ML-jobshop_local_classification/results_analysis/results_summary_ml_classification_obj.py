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
    #
    instances = [20, 40, 60, 80, 100]
    sizeStr = ["20x20", "40x40", "60x60", "80x80", "100x100"]

    # instances = [20, 40, 60]
    # sizeStr = ["20x20", "40x40", "60x60"]

    # instances = [6, 8]
    # sizeStr = ["6x6", "8x8"]

    # methods = ["NONE_local_classification", "GP_local_classification", "Optimal_global_regression"]
    # methods = ["NONE_global_regression", "GP_global_regression", "Optimal_global_regression"]

    # methods = ["NONE_global_regression", "GP_global_regression", "Optimal_global_regression", "GPc_local_classification",
    #            "LR_local_classification", "SVM_local_classification", "KNN_local_classification", "DT_local_classification",
    #            "MLP_local_classification", "MLP_100_local_classification"]
    # methodStr = ["Default", "BranGP", "OptStart", "ClasGP", "LR", "SVM", "KNN", "DT", "MLP_1024", "MLP_100"]

    # methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
    #            "GP_global_regression", "GP_pure_classification","GPc_local_classification", "GPc_pure_classification",
    #            "SVM_local_classification", "SVM_pure_classification"]
    # methodStr = ["Default", "MinDom", "LowMin", "BranGP", "BranGP_p", "ClasGP", "ClasGP_p",  "SVM", "SVM_p"]

    # methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
    #            "GP_global_regression", "GPc_local_classification", "SVM_local_classification", "LR_local_classification",
    #            "MLP_100_local_classification",]
    # methodStr = ["Default", "MinDom", "LowMin", "BranGP", "ClasGP", "SVM", "LR", "MLP"]

    # methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
    #            "GP_global_regression", "GPc_local_classification", "SVM_local_classification", "LR_local_classification",
    #            "MLP_100_local_classification"]
    # methodStr = ["Default", "MinDom", "LowMin", "BranGP", "ClasGP", "SVM", "LR", "MLP"]

    methods = ["NONE_global_regression", "MIN_DOMAIN_SIZE_local_classification", "LOWEST_MIN_local_classification",
               "GP_global_regression", "GPc_local_classification", "MLP_100_local_classification", "SVM_local_classification"]
    methodStr = ["Default", "MinDom", "LowMin", "BranGP-H", "GP-H", "MLP-H", "SVM-H"]

    # methods = ["NONE_global_regression", "GP_global_regression", "GPc_local_classification", "SVM_local_classification"]
    # methodStr = ["Default", "BranGP", "ClasGP", "SVM"]

    # methods = ["NONE_global_regression", "GP_global_regression", "Optimal_global_regression", "GPc_local_classification", "LR_local_classification"]
    # methodStr = ["Default", "BranGP", "OptStart", "ClasGP", "LR"]

    objs = []
    branches = []
    runtimes = []
    solved = []


    for instance in instances:
        for ml_method in methods:
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
                # imp.append((branches[id2,k] - branches[id1,k])/max(branches[id2,k],branches[id1,k]))
                imp.append((objs[id2, k] - objs[id1, k]) / max(objs[id2, k], objs[id1, k]))
                # imp.append((branches[id2, k] - branches[id1, k]) / branches[id2, k])

    # imp = np.array(imp).reshape(len(instances) * (len(methods)-1), 100)
    imp = np.array(imp).reshape(len(instances) * (len(methods)), 100)
    # imp = np.transpose(imp)



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

    ax.legend(loc='lower left', fontsize=17, ncol=3)
    plt.xlabel('Instance Size', fontsize=20)
    plt.ylabel('Improvement', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    filename = 'figs/ml_obj_' + obj + '_imp.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

    table = []
    ave = []
    std = []
    pvalues = []


    idx = 0
    max_val = 0
    for j in range(1, len(methods)):
        val = df.Improvement[df.Method == methodStr[j]]
        if val.mean() > max_val:
            max_val = val.mean()
            idx = j

    val1 = df.Improvement[df.Method == methodStr[1]]
    for j in range(1, len(methods)):
        val2 = df.Improvement[df.Method == methodStr[j]]
        (statistic, pvalue) = stats.ttest_rel(val1, val2)
        if j == idx:
            ave.append(str('\\bf{') + "{:.3f}".format(val2.mean()) + "}")
        else:
            ave.append("{:.3f}".format(val2.mean()))
        std.append("{:.3f}".format(val2.std()))
        if pvalue >= 0.05:
            pvalues.append("{:.0e}".format(pvalue))
        else:
            pvalues.append("\\emph{" + "{:.0e}".format(pvalue) + "}")

        print(val1.mean(), val2.mean(), "{:.0e}".format(pvalue))

    table.append(methodStr[1:len(methods)])
    table.append(ave)
    table.append(std)
    table.append(pvalues)

    table = np.array(table).reshape(4, len(methods) - 1)

    print(table)
    np.savetxt("results_summary/ml_obj_{}_table.csv".format(obj), table, delimiter=",", fmt='%s')


    # filename = 'figs/optimal_' + obj + '_imp.png'
    # plt.savefig(filename, bbox_inches='tight')

    # np.savetxt("cmax_optimal_{}_ave_objs.csv".format(level), np.transpose(objs), delimiter=",")
    # np.savetxt("cmax_optimal_{}_ave_branches.csv".format(level), np.transpose(branches), delimiter=",")
    # np.savetxt("cmax_optimal_{}_ave_runtimes.csv".format(level), np.transpose(runtimes), delimiter=",")

    #
    # print('==================================')
    # print("Average running time --- %s seconds ---" % alltime.mean())
    # print("Average number of branches --- %s ---" % allbranches.mean())
    # print("Average objective values --- %s ---" % allobjs.mean())

