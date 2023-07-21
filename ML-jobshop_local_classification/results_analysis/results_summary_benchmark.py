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

    files = ['la01', 'la07', 'la05', 'abz7', 'abz8', 'la04', 'abz6', 'la03', 'la08', 'la02', 'abz9', 'la06', 'abz5',
             'la20', 'la22', 'la30', 'la16', 'la24', 'la19', 'la29', 'la10', 'la21', 'la11', 'la15', 'la17', 'la09',
             'la28', 'la26', 'la12', 'la18', 'la25', 'la14', 'la27', 'la23', 'la31', 'la13', 'la36', 'orb04', 'orb02',
             'orb10', 'orb08', 'orb07', 'orb09', 'la38', 'orb06', 'la39', 'orb05', 'la40', 'la32', 'swv02', 'orb03',
             'la33', 'la37', 'orb01', 'la34', 'swv01', 'la35', 'swv04', 'ta03', 'ta01', 'swv17', 'swv14', 'ta05',
             'swv15', 'swv06', 'swv20', 'swv03', 'swv11', 'ta02', 'swv16', 'swv09', 'swv07', 'swv08', 'swv19', 'swv18',
             'swv13', 'swv12', 'swv05', 'ta04', 'swv10', 'ta23', 'ta27', 'ta25', 'ta15', 'ta06', 'ta19', 'ta28', 'ta13',
             'ta21', 'ta17', 'ta26', 'ta18', 'ta20', 'ta12', 'ta07', 'ta08', 'ta14', 'ta10', 'ta24', 'ta16', 'ta09',
             'ta22', 'ta11', 'ta42', 'ta46', 'ta39', 'ta33', 'ta47', 'ta41', 'ta38', 'ta50', 'ta34', 'ta31', 'ta43',
             'ta37', 'ta48', 'ta40', 'ta30', 'ta35', 'ta49', 'ta36', 'ta44', 'ta29', 'ta32', 'ta45', 'ta68', 'ta62',
             'ta54', 'ta69', 'ta51', 'ta52', 'ta59', 'ta57', 'ta60', 'ta61', 'ta58', 'ta70', 'ta65', 'ta64', 'ta71',
             'ta56', 'ta66', 'ta53', 'ta67', 'ta63', 'ta55', 'ta80', 'ta75', 'ta74', 'ta77', 'ta72', 'ta79', 'ta73',
             'ta76', 'ta78']  # benchmark instances for cmax

    # ml_methods = ["NONE_local", "GP_local", "SVM_local", "DT_local", "SVM_regression", "MLP_1024_regression","MLP_512_regression",
    #               "MLP_200_regression", "MLP_100_regression", "MLP_50_regression", "OLS_regression", "DT_regression",
    #               "KNN_regression"]
    # ml_methods = ["NONE_local", "GP_local", "DT_local", "SVM_local", "MLP_2048_regression", "MLP_1024_regression",
    #               "MLP_512_regression", "MLP_200_regression", "MLP_100_regression", "MLP_50_regression"]
    ml_methods = ["NONE_local", "GP_local", "SVM_regression", "DT_local", "MLP_1024_regression"]
    cutoff = 60


    solved = []

    objs = []
    runtimes = []
    branches = []

    for method in ml_methods:
        df_objs = np.genfromtxt("../results_old/benchmark_{}_{}_objs.csv".format(obj, method),delimiter=",")
        df_branches = np.genfromtxt("../results_old/benchmark_{}_{}_branches.csv".format(obj, method),delimiter=",")
        df_runtimes = np.genfromtxt("../results_old/benchmark_{}_{}_runtimes.csv".format(obj, method),delimiter=",")

        objs.append(df_objs)
        runtimes.append(df_runtimes)
        branches.append(df_branches)
        solved.append(sum(time < cutoff - 0.1 for time in df_runtimes))


    objs = np.transpose(objs)
    runtimes = np.transpose(runtimes)

    # save table
    table = []
    k = 0
    for i in range(len(files)):
        if runtimes[i, 0] >= cutoff:
            k = k + 1
            table.append(files[i])
            for j in range(len(ml_methods)):
                table.append(objs[i, j])
            for j in range(len(ml_methods)):
                table.append(runtimes[i, j])

    table = np.array(table).reshape(k,2*len(ml_methods)+1)

    print(table)

    np.savetxt("results_summary/benchmark_{}_table.csv".format(obj), np.array(table), delimiter=",", fmt='%s')


    #
    # np.savetxt("/results_summary/benchmark_{}_all_objs.csv".format(obj), np.transpose(objs_all), delimiter=",")
    # np.savetxt("/results_summary/benchmark_{}_all_runtimes.csv".format(obj), np.transpose(runtimes_all), delimiter=",")



