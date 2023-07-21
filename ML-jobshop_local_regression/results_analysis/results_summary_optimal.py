from __future__ import print_function
import collections
import numpy as np
import random
import argparse
import pandas as pd

if __name__=='__main__':

    # instances = [10, 11, 12, 13, 14]
    instances = range(5, 15)

    level = "global"

    objs = []
    branches = []
    runtimes = []
    solved = []

    for instance in instances:
        df_objs = np.genfromtxt("../results/cmax_optimal_{}_nm_{}_nj_{}_objs.csv".format(level, instance, instance),delimiter=",")
        df_branches = np.genfromtxt("../results/cmax_optimal_{}_nm_{}_nj_{}_branches.csv".format(level, instance, instance),delimiter=",")
        df_runtimes = np.genfromtxt("../results/cmax_optimal_{}_nm_{}_nj_{}_runtimes.csv".format(level, instance, instance),delimiter=",")

        objs.append(df_objs[:,0].mean())
        objs.append(df_objs[:,1].mean())
        branches.append(df_branches[:,0].mean())
        branches.append(df_branches[:,1].mean())
        runtimes.append(df_runtimes[:,0].mean())
        runtimes.append(df_runtimes[:,1].mean())


    objs = np.array(objs).reshape(len(instances),2)
    branches = np.array(branches).reshape(len(instances),2)
    runtimes = np.array(runtimes).reshape(len(instances),2)

    np.savetxt("cmax_optimal_{}_ave_objs.csv".format(level), np.transpose(objs), delimiter=",")
    np.savetxt("cmax_optimal_{}_ave_branches.csv".format(level), np.transpose(branches), delimiter=",")
    np.savetxt("cmax_optimal_{}_ave_runtimes.csv".format(level), np.transpose(runtimes), delimiter=",")

    # allobjs = np.array(allobjs)
    # allbranches = np.array(allbranches)
    # alltime = np.array(alltime)

    # np.savetxt("{}_nm_{}_nj_{}_objs.csv".format(ml_method, nmachine, njob), allobjs, delimiter=",")
    # np.savetxt("{}_nm_{}_nj_{}_branches.csv".format(ml_method, nmachine, njob), allbranches, delimiter=",")
    # np.savetxt("{}_nm_{}_nj_{}_runtimes.csv".format(ml_method, nmachine, njob), alltime, delimiter=",")
    #
    # print('==================================')
    # print("Average running time --- %s seconds ---" % alltime.mean())
    # print("Average number of branches --- %s ---" % allbranches.mean())
    # print("Average objective values --- %s ---" % allobjs.mean())

