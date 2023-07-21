from __future__ import print_function
import collections
import numpy as np
import random
import argparse
import pandas as pd
from scipy.stats import ttest_ind

from scipy import stats

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='testing.')
    parser.add_argument('-nmachine', default=10, type=int, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-obj', default="cmax", type=str, help='objective function')  # "cmax", "twt" "tmax"
    parser.add_argument('-idx', default=0, type=int, help='the index of dataset')
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
    idx = args.idx

    ml_methods = ["NONE", "MIN_DOMAIN_SIZE", "LOWEST_MIN", "GP", "GPc", "SVM", "MLP_100"]
    methodStr = ["Default", "MinDom", "LowMin", "BranGP", "GP-H", "SVM-H", "MLP-H"]

    cutoff = 600

    # datasets = ["ta", "la", "abz", "orb", "swv"]
    datasets = ["ta", "la", "abz", "orb", "swv"]

    table = []

    header = []
    header.append("Benchmark")
    header.append("\#Instance")
    header.append("Stats")
    for meth in methodStr:
        header.append(meth)

    table.append(header)

    imp_all = []
    # table.append(methodStr[1:len(ml_methods)])
    acc_all = 0
    for dataset in datasets:

        if dataset == "ta":
            # 80 ta instances
            files = ['ta01', 'ta02', 'ta03', 'ta04', 'ta05', 'ta06', 'ta07', 'ta08', 'ta09', 'ta10', 'ta11', 'ta12', 'ta13',
                     'ta14', 'ta15', 'ta16', 'ta17', 'ta18', 'ta19', 'ta20', 'ta21', 'ta22', 'ta23', 'ta24', 'ta25', 'ta26',
                     'ta27', 'ta28', 'ta29', 'ta30', 'ta31', 'ta32', 'ta33', 'ta34', 'ta35', 'ta36', 'ta37', 'ta38', 'ta39',
                     'ta40', 'ta41', 'ta42', 'ta43', 'ta44', 'ta45', 'ta46', 'ta47', 'ta48', 'ta49', 'ta50', 'ta51', 'ta52',
                     'ta53', 'ta54', 'ta55', 'ta56', 'ta57', 'ta58', 'ta59', 'ta60', 'ta61', 'ta62', 'ta63', 'ta64', 'ta65',
                     'ta66', 'ta67', 'ta68', 'ta69', 'ta70', 'ta71', 'ta72', 'ta73', 'ta74', 'ta75', 'ta76', 'ta77', 'ta78',
                     'ta79', 'ta80']
        elif dataset == "la":
            # 40 la instances
            files = ['la01', 'la02', 'la03', 'la04', 'la05', 'la06', 'la07', 'la08', 'la09', 'la10', 'la11', 'la12', 'la13',
                     'la14', 'la15', 'la16', 'la17', 'la18', 'la19', 'la20', 'la21', 'la22', 'la23', 'la24', 'la25', 'la26',
                     'la27', 'la28', 'la29', 'la30', 'la31', 'la32', 'la33', 'la34', 'la35', 'la36', 'la37', 'la38', 'la39',
                     'la40']
        elif dataset == "abz":
            # 5 abz instances
            files = ['abz5', 'abz6', 'abz7', 'abz8', 'abz9']
        elif dataset == "orb":
            # 10 orb instances
            files = ['orb01', 'orb02', 'orb03', 'orb04', 'orb05', 'orb06', 'orb07', 'orb08', 'orb09', 'orb10']
        elif dataset == "swv":
            # 20 swv instances
            files = ['swv01', 'swv02', 'swv03', 'swv04', 'swv05', 'swv06', 'swv07', 'swv08', 'swv09', 'swv10', 'swv11',
                     'swv12', 'swv13', 'swv14', 'swv15', 'swv16', 'swv17', 'swv18', 'swv19', 'swv20']


        solved = []
        solved.append("\multirow{" + "3}" + "{*}" + "{" + dataset + "}")
        solved.append("\multirow{" + "3}" + "{*}" + "{" + "{}".format(len(files)) + "}")
        solved.append("\#Solved")

        objs = []
        runtimes = []
        branches = []
        for method in ml_methods:
            if method == "MIN_DOMAIN_SIZE" or method == "LOWEST_MIN":
                df_objs = np.genfromtxt("../results_cmax/benchmark_{}_{}_{}_objs.csv".format(obj, method, dataset),delimiter=",")
                df_branches = np.genfromtxt("../results_cmax/benchmark_{}_{}_{}_branches.csv".format(obj, method, dataset),delimiter=",")
                df_runtimes = np.genfromtxt("../results_cmax/benchmark_{}_{}_{}_runtimes.csv".format(obj, method, dataset),delimiter=",")
            else:
                df_objs = np.genfromtxt("../results_cmax_hybrid/benchmark_{}_{}_{}_objs.csv".format(obj, method, dataset),
                                        delimiter=",")
                df_branches = np.genfromtxt(
                    "../results_cmax_hybrid/benchmark_{}_{}_{}_branches.csv".format(obj, method, dataset), delimiter=",")
                df_runtimes = np.genfromtxt(
                    "../results_cmax_hybrid/benchmark_{}_{}_{}_runtimes.csv".format(obj, method, dataset), delimiter=",")

            objs.append(df_objs)
            runtimes.append(df_runtimes)
            branches.append(df_branches)
            solved.append("{:.0f}".format(sum(time < cutoff for time in df_runtimes)))


        objs = np.transpose(objs)
        runtimes = np.transpose(runtimes)
        branches = np.transpose(branches)


        table.append(solved)

        print(solved)

        unsolved_objs = []
        solved_branches = []
        acc1 = 0
        acc2 = 0
        for i in range(len(files)):
            if runtimes[i,0] < cutoff:
                acc1 += 1
                for j in range(0,len(ml_methods)):
                    solved_branches.append(branches[i, j])
            else:
                acc2 += 1
                for j in range(0,len(ml_methods)):
                    unsolved_objs.append(objs[i, j])

        solved_branches = np.array(solved_branches).reshape(acc1, len(ml_methods))
        unsolved_objs = np.array(unsolved_objs).reshape(acc2, len(ml_methods))

        print(solved_branches)
        print(unsolved_objs)

        ave_objs = []
        ave_bran = []

        ave_objs.append("spaceX")
        ave_objs.append("spaceX")
        ave_objs.append("ObjVal")

        ave_bran.append("spaceX")
        ave_bran.append("spaceX")
        ave_bran.append("Branch")

        idx = 0
        min_val = unsolved_objs[:,0].mean()
        for j in range(1, len(ml_methods)):
            if unsolved_objs[:,j].mean() < min_val:
                min_val = unsolved_objs[:,j].mean()
                idx = j

        for j in range(0,len(ml_methods)):
            if j == idx:
                ave_objs.append(str('\\bf{') + "{:.2e}".format(unsolved_objs[:, j].mean()) + "}")
            else:
                ave_objs.append("{:.2e}".format(unsolved_objs[:,j].mean()))

        idx = 0
        min_val = solved_branches[:, 0].mean()
        for j in range(1, len(ml_methods)):
            if solved_branches[:, j].mean() < min_val:
                min_val = solved_branches[:, j].mean()
                idx = j

        for j in range(0, len(ml_methods)):
            if j == idx:
                ave_bran.append(str('\\bf{') + "{:.2e}".format(solved_branches[:, j].mean()) + "}")
            else:
                ave_bran.append("{:.2e}".format(solved_branches[:, j].mean()))

        print(ave_objs)
        print(ave_bran)

        table.append(ave_bran)
        table.append(ave_objs)

    table = np.array(table).reshape(3*len(datasets)+1, len(ml_methods)+3)

    print(table)


    np.savetxt("results_summary/ml_benchmark_{}_table.csv".format(obj), table, delimiter=",", fmt='%s')



    # np.savetxt("results_summary/benchmark_{}_table.csv".format(obj), np.array(table), delimiter=",", fmt='%s')


    #
    # np.savetxt("/results_summary/benchmark_{}_all_objs.csv".format(obj), np.transpose(objs_all), delimiter=",")
    # np.savetxt("/results_summary/benchmark_{}_all_runtimes.csv".format(obj), np.transpose(runtimes_all), delimiter=",")



