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
    parser.add_argument('-obj', default="twt", type=str, help='objective function')  # "cmax", "twt" "tmax"
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

    files = ['abz5', 'abz6', 'la16', 'la17', 'la18', 'la19', 'la20', 'orb01', 'orb02', 'orb03', 'orb04', 'orb05',
             'orb06', 'orb07', 'orb08', 'orb09', 'orb10']  # benchmark instances for twt

    # ml_methods = ["NONE", "GP", "GPc", "LR", "SVM", "MLP_100"]
    # methodStr = ["Default", "BranGP", "ClasGP", "LR", "SVM", "MLP"]

    ml_methods = ["NONE", "MIN_DOMAIN_SIZE", "LOWEST_MIN", "GP", "GPc", "SVM", "MLP_100"]
    methodStr = ["Default", "MinDom", "LowMin", "BranGP", "GP-H", "SVM-H", "MLP-H"]

    cutoff = 600


    solved = []

    objs = []
    runtimes = []
    branches = []

    for method in ml_methods:
        if method == "MIN_DOMAIN_SIZE" or method == "LOWEST_MIN":
            df_objs = np.genfromtxt("../results_twt/benchmark_{}_{}_objs.csv".format(obj, method),delimiter=",")
            df_branches = np.genfromtxt("../results_twt/benchmark_{}_{}_branches.csv".format(obj, method),delimiter=",")
            df_runtimes = np.genfromtxt("../results_twt/benchmark_{}_{}_runtimes.csv".format(obj, method),delimiter=",")
        else:
            df_objs = np.genfromtxt("../results_twt_hybrid/benchmark_{}_{}_objs.csv".format(obj, method), delimiter=",")
            df_branches = np.genfromtxt("../results_twt_hybrid/benchmark_{}_{}_branches.csv".format(obj, method),
                                        delimiter=",")
            df_runtimes = np.genfromtxt("../results_twt_hybrid/benchmark_{}_{}_runtimes.csv".format(obj, method),
                                        delimiter=",")


        objs.append(df_objs)
        runtimes.append(df_runtimes)
        branches.append(df_branches)
        solved.append(sum(time < cutoff - 0.1 for time in df_runtimes))


    objs = np.transpose(objs)
    runtimes = np.transpose(runtimes)
    branches = np.transpose(branches)

    table = []
    table.append("Instance")
    table.append("Stats")
    for meth in methodStr:
        table.append(meth)


    solved_datasets = []
    for i in range(len(files)):
        if runtimes[i,0] < cutoff:
            solved_datasets.append(files[i])
            table.append(files[i])
            table.append("branch")


            idx = 0
            minVal = branches[i, 0]
            for j in range(0,len(ml_methods)):
                if minVal > branches[i, j]:
                    minVal = branches[i, j]
                    idx = j


            for j in range(0,len(ml_methods)):
                if j == idx:
                    table.append(str('\\bf{') + "{:.2e}".format(branches[i,j]) + "}")
                else:
                    table.append("{:.2e}".format(branches[i, j]))

    print(solved_datasets)

    for i in range(len(files)):
        if runtimes[i,0] >= cutoff:
            table.append(files[i])
            table.append("objVal")
            idx = 0
            minVal = objs[i, 0]
            for j in range(0, len(ml_methods)):
                if minVal > objs[i, j]:
                    minVal = objs[i, j]
                    idx = j

            for j in range(0, len(ml_methods)):
                if j == idx:
                    table.append(str('\\bf{') + "{:.2e}".format(objs[i, j]) + "}")
                else:
                    table.append("{:.2e}".format(objs[i, j]))


    table= np.array(table).reshape(len(files)+1, len(ml_methods)+2)

    print(table)

    np.savetxt("results_summary/ml_benchmark_twt_table.csv", table, delimiter=",", fmt='%s')


    # print(objs)
    # print(runtimes)
    # print(branches)
    # print(solved)


    # for i in range(len(ml_methods)):
    #     print(objs[:,i].mean(), runtimes[:,i].mean(), branches[:,i].mean())

    # # save table
    # table = []
    # k = 0
    # for i in range(len(files)):
    #     if runtimes[i, 0] >= cutoff:
    #         k = k + 1
    #         table.append(files[i])
    #         for j in range(len(ml_methods)):
    #             table.append(objs[i, j])
    #         for j in range(len(ml_methods)):
    #             table.append(runtimes[i, j])
    #
    # table = np.array(table).reshape(k,2*len(ml_methods)+1)
    #
    # print(table)

    # np.savetxt("results_summary/benchmark_{}_table.csv".format(obj), np.array(table), delimiter=",", fmt='%s')


    #
    # np.savetxt("/results_summary/benchmark_{}_all_objs.csv".format(obj), np.transpose(objs_all), delimiter=",")
    # np.savetxt("/results_summary/benchmark_{}_all_runtimes.csv".format(obj), np.transpose(runtimes_all), delimiter=",")



