from __future__ import print_function
import collections
from ortools.sat.python import cp_model
import numpy as np
import random
import argparse
import pandas as pd
import joblib
import time
from operator import itemgetter, attrgetter

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, obj, opt_val, M):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__optval = opt_val
        self.__obj = obj
        self.__sequence = 1
        self.__M = M

    def on_solution_callback(self):
        self.__solution_count += 1
        s = []
        if self.Value(self.__obj) == self.__optval:
            for v in self.__variables:
                print('%s=%i' % (v[0], self.Value(v[0])), end=' ')
                s.append(self.Value(v[0]))
            sortedid = sorted(range(len(s)), key=lambda k: s[k])
            # seq = [str(self.__variables[i][0]) for i in sortedid]
            # seqM = [[] for _ in range(self.__M)]
            # for i in sortedid:
            #     seqM[self.__variables[i][1]].append(i)
            self.__sequence = 2
            print()

    def solution_count(self):
        return self.__solution_count


def solve_jss_ortools(nmachines=3, njobs=3, seed=1, obj="tmax", h=1.3, level="global", ml_model="DT", cutoff=100, data_type="raw"):
    random.seed(seed)
    """Minimal jobshop problem."""

    N = njobs
    M = nmachines
    jobs_data = []
    release = {}
    duedate = {}
    weight = {}
    totaljobduration = []
    for i in range(N):
        nops = M  # random.randint(1, M)
        jorder = list(range(M))
        random.shuffle(jorder)
        jorder = jorder[:nops]
        jtime = [random.randint(1,50) for _ in range(nops)]
        totaldur = sum([jtime[j] for j in range(nops)])
        totaltime = np.sum(jtime)
        release[i] = random.randint(0, int(totaltime / 2))
        weight[i] = random.choices([1, 2, 4], [0.2, 0.6, 0.2])[0]
        duedate[i] = int(release[i] + h * totaltime)
        totaljobduration.append(totaldur)
        job = [(jorder[j],
                jtime[j],
                sum([jtime[k] for k in range(j)]),
                sum([jtime[k] for k in range(j, nops)])) for j in range(nops)]
        jobs_data.append(job)
    # print(jobs_data)

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    jobs_count = len(jobs_data)
    all_jobs = range(jobs_count)

    # Compute horizon.
    horizon = sum(task[1] for job in jobs_data for task in job)

    task_type = collections.namedtuple('task_type', 'start end interval')
    assigned_task_type = collections.namedtuple('assigned_task_type', 'start job index')


    optimal_obj = None
    # Create the model.
    model = cp_model.CpModel()
    # Create jobs.
    all_tasks = {}
    list_of_vars = []
    varlist = []
    feature_dict = {}
    count = 0
    for job in all_jobs:
        total_dur = 0
        earliest_start = release[job]
        matched_varlist = []
        for task_id, task in enumerate(jobs_data[job]):
            start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, task_id))
            duration = task[1]
            end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
            all_tasks[job, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            list_of_vars.append((all_tasks[job, task_id].end, task[0], (job, task_id), count))
            varlist.append(start_var)
            feature_dict[start_var] = {}  # [task_id, earliest_start, duration, 0, 0,  0]  # task_id, earliest_start, duration, workload, sum_early
            feature_dict[start_var]["taskid"] = task_id
            feature_dict[start_var]["earliest_start"] = earliest_start
            feature_dict[start_var]["duration"] = duration
            feature_dict[start_var]["weight"] = weight[job]
            feature_dict[start_var]["duedate"] = duedate[job]
            feature_dict[start_var]["release"] = release[job]
            earliest_start += duration
            matched_varlist.append(start_var)
            total_dur += duration
            for var in matched_varlist:
                feature_dict[var]["total_dur"] = total_dur  # - feature_dict[var][1]
            count += 1

    # Create and add disjunctive constraints.
    maxload = 0
    for machine in all_machines:
        intervals = []
        matched_varlist = []
        workload = 0
        sum_early_start = 0
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                if task[0] == machine:
                    intervals.append(all_tasks[job, task_id].interval)
                    matched_varlist.append(all_tasks[job, task_id].start)
                    workload += task[1]
                    sum_early_start += feature_dict[all_tasks[job, task_id].start]["earliest_start"]
        model.AddNoOverlap(intervals)
        for var in matched_varlist:
            feature_dict[var]["workload"] = workload
            maxload = max(workload, maxload)
    for var in feature_dict:
        feature_dict[var]["maxload"] = maxload

    # optimal_obj = None
    # # Create the model.
    # model = cp_model.CpModel()
    # # Create jobs.
    # all_tasks = {}
    # list_of_vars = []
    # count = 0
    # for job in all_jobs:
    #     for task_id, task in enumerate(jobs_data[job]):
    #         start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, task_id))
    #         duration = task[1]
    #         end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
    #         interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
    #         all_tasks[job, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
    #         list_of_vars.append((all_tasks[job, task_id].end, task[0], (job, task_id), count))
    #         count += 1
    #
    # # Create and add disjunctive constraints.
    # for machine in all_machines:
    #     intervals = []
    #     for job in all_jobs:
    #         for task_id, task in enumerate(jobs_data[job]):
    #             if task[0] == machine:
    #                 intervals.append(all_tasks[job, task_id].interval)
    #     model.AddNoOverlap(intervals)


    # Add precedence contraints.
    for job in all_jobs:
        for task_id in range(0, len(jobs_data[job]) - 1):
            model.Add(all_tasks[job, task_id + 1].start >= all_tasks[job, task_id].end)

    # order decision variables by optimal solution predictions
    ml_start_time = time.time()
    if ml_model != "NONE" and ml_model != "GP":
        # compute features
        disjunctive_raw_features = []
        disjunctive_combine_features = []
        if level == "global":
            for var1 in list_of_vars:
                (job1, task_id1) = var1[2]
                count1 = var1[3]
                for var2 in list_of_vars:
                    (job2, task_id2) = var2[2]
                    count2 = var2[3]
                    if count1 < count2:
                        disjunctive_raw_features.append([
                            task_id1, task_id2,
                            jobs_data[job1][task_id1][1], jobs_data[job2][task_id2][1],
                            jobs_data[job1][task_id1][2], jobs_data[job2][task_id2][2],
                            jobs_data[job1][task_id1][3], jobs_data[job2][task_id2][3],
                            totaljobduration[job1], totaljobduration[job2],
                            duedate[job1], duedate[job2],
                            weight[job1], weight[job2],
                            release[job1], release[job2],
                            feature_dict[all_tasks[job1, task_id1].start]["workload"], feature_dict[all_tasks[job2, task_id2].start]["workload"],
                            feature_dict[all_tasks[job1, task_id1].start]["maxload"], feature_dict[all_tasks[job2, task_id2].start]["maxload"],
                        ])

                        disjunctive_combine_features.append([
                            task_id1 - task_id2,
                            jobs_data[job1][task_id1][1] - jobs_data[job2][task_id2][1],
                            jobs_data[job1][task_id1][2] - jobs_data[job2][task_id2][2],
                            jobs_data[job1][task_id1][3] - jobs_data[job2][task_id2][3],
                            totaljobduration[job1] - totaljobduration[job2],
                            duedate[job1] - duedate[job2],
                            weight[job1] - weight[job2],
                            release[job1] - release[job2],
                            feature_dict[all_tasks[job1, task_id1].start]["workload"] - feature_dict[all_tasks[job2, task_id2].start]["workload"],
                            feature_dict[all_tasks[job1, task_id1].start]["maxload"] - feature_dict[all_tasks[job2, task_id2].start]["maxload"],
                        ])
        elif level == "local":
            # Create one list of assigned tasks per machine.
            jobs_per_machine = [[] for _ in all_machines]
            for job in all_jobs:
                for task_id, task in enumerate(jobs_data[job]):
                    machine = task[0]
                    jobs_per_machine[machine].append(assigned_task_type(start=0,job=job,index=task_id))
            for machine in all_machines:
                operation_raw_features = []
                for assigned_task in jobs_per_machine[machine]:
                    operation_raw_features.append([assigned_task.index,  # operation index
                                                   jobs_data[assigned_task.job][assigned_task.index][1],# operation duration
                                                   jobs_data[assigned_task.job][assigned_task.index][2],# earliest start time (total upstream durations)
                                                   jobs_data[assigned_task.job][assigned_task.index][3],# earliest completion time
                                                   totaljobduration[assigned_task.job],# total duration of job the operation belong to
                                                   duedate[assigned_task.job],
                                                   weight[assigned_task.job],
                                                   release[assigned_task.job],
                                                   ])

                for i in range(len(operation_raw_features)):
                    for j in range(len(operation_raw_features)):
                        if i < j:
                            disjunctive_raw_features.append([
                                operation_raw_features[i][0], operation_raw_features[j][0],
                                operation_raw_features[i][1], operation_raw_features[j][1],
                                operation_raw_features[i][2], operation_raw_features[j][2],
                                operation_raw_features[i][3], operation_raw_features[j][3],
                                operation_raw_features[i][4], operation_raw_features[j][4],
                                operation_raw_features[i][5], operation_raw_features[j][5],
                                operation_raw_features[i][6], operation_raw_features[j][6],
                                operation_raw_features[i][7], operation_raw_features[j][7],
                            ])

                            disjunctive_combine_features.append([
                                operation_raw_features[i][0] - operation_raw_features[j][0],
                                operation_raw_features[i][1] - operation_raw_features[j][1],
                                operation_raw_features[i][2] - operation_raw_features[j][2],
                                operation_raw_features[i][3] - operation_raw_features[j][3],
                                operation_raw_features[i][4] - operation_raw_features[j][4],
                                operation_raw_features[i][5] - operation_raw_features[j][5],
                                operation_raw_features[i][6] - operation_raw_features[j][6],
                                operation_raw_features[i][7] - operation_raw_features[j][7],
                            ])

        # compute class labels for each pair of operations
        if ml_model == "Hybrid":
            trained_ml = joblib.load("../training_data/{}_SVM_{}_{}.pkl".format(obj, level, data_type))
        else:
            trained_ml = joblib.load("../training_data/{}_{}_{}_{}.pkl".format(obj, ml_method, level, data_type))

        if data_type == "combine":
            predicted_label = trained_ml.predict(np.array(disjunctive_combine_features))
        elif data_type == "raw":
            predicted_label = trained_ml.predict(np.array(disjunctive_raw_features))

        # compute the priorties of decision variables based on ML prediction
        var_priorities_ml = {}
        for var in list_of_vars:
            (job, task_id) = var[2]
            var_priorities_ml[all_tasks[job,task_id].start] = 0

        if level == "global":
            idx = 0
            for var1 in list_of_vars:
                (job1, task_id1) = var1[2]
                count1 = var1[3]
                for var2 in list_of_vars:
                    (job2, task_id2) = var2[2]
                    count2 = var2[3]
                    if count1 < count2:
                        if predicted_label[idx] == 1:
                            var_priorities_ml[all_tasks[job1,task_id1].start] += 1
                        elif predicted_label[idx] == -1:
                            var_priorities_ml[all_tasks[job2,task_id2].start] += 1
                        idx = idx + 1
        elif level == "local":
            idx = 0
            for machine in all_machines:
                for i in range(len(jobs_per_machine[machine])):
                    for j in range(len(jobs_per_machine[machine])):
                        if i < j:
                            assigned_task1 = jobs_per_machine[machine][i]
                            assigned_task2 = jobs_per_machine[machine][j]
                            if predicted_label[idx] == 1:
                                var_priorities_ml[all_tasks[assigned_task1.job, assigned_task1.index].start] += 1
                            else:
                                var_priorities_ml[all_tasks[assigned_task2.job, assigned_task2.index].start] += 1
                            idx = idx + 1

        if ml_model != "Hybrid":
            # sort variables in order based on the priorties
            var_select_order = [x for _, x in sorted([(-var_priorities_ml[v], v) for v in var_priorities_ml], key=itemgetter(0))]
            model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

    if ml_model == "GP" or ml_model == "Hybrid":
        var_priorities_gp = {}
        for var in list_of_vars:
            (job, task_id) = var[2]
            var_priorities_gp[all_tasks[job, task_id].start] = 0
            feature_inputs = [
                feature_dict[all_tasks[job, task_id].start]["earliest_start"],
                feature_dict[all_tasks[job, task_id].start]["duration"],
                feature_dict[all_tasks[job, task_id].start]["weight"],
                feature_dict[all_tasks[job, task_id].start]["duedate"],
                feature_dict[all_tasks[job, task_id].start]["release"],
                feature_dict[all_tasks[job, task_id].start]["total_dur"],
                feature_dict[all_tasks[job, task_id].start]["workload"],
                feature_dict[all_tasks[job, task_id].start]["maxload"]
            ]
            ES, PT, W, DD, RL, TPT, WL, MaxWL = feature_inputs[0], feature_inputs[1], \
                                                feature_inputs[2], feature_inputs[3], feature_inputs[4], feature_inputs[
                                                    5], feature_inputs[6], feature_inputs[7]
            if obj == "cmax":
                out = -((DD + RL) - 2 * ES)
            elif obj == "tmax":
                out = TPT * WL * ES
            elif obj == "twt":
                out = -(W - PT - WL - ES)
            var_priorities_gp[all_tasks[job, task_id].start] = out
        if ml_method == "GP":
            # sort variables in order based on the priorties
            var_select_order = [x for _, x in
                                sorted([(var_priorities_gp[v], v) for v in var_priorities_gp], key=itemgetter(0))]
            model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

    if ml_model == "Hybrid":
        var_priorities = {}

        min_ml = min(var_priorities_ml[v] for v in var_priorities_ml)
        max_ml = max(var_priorities_ml[v] for v in var_priorities_ml)
        min_gp = min(var_priorities_gp[v] for v in var_priorities_gp)
        max_gp = max(var_priorities_gp[v] for v in var_priorities_gp)
        for var in list_of_vars:
            (job, task_id) = var[2]
            var_priorities_ml[all_tasks[job, task_id].start] = (float(var_priorities_ml[all_tasks[job, task_id].start]) - min_ml) / (min_ml - max_ml)
            var_priorities_gp[all_tasks[job, task_id].start] = (float(
                var_priorities_gp[all_tasks[job, task_id].start]) - min_gp) / (max_gp - min_gp)

        for var in list_of_vars:
            (job, task_id) = var[2]
            var_priorities[all_tasks[job, task_id].start] = 0.99*var_priorities_ml[all_tasks[job, task_id].start]+ 0.01*var_priorities_gp[all_tasks[job, task_id].start]
        # sort variables in order based on the priorties
        var_select_order = [x for _, x in sorted([(var_priorities[v], v) for v in var_priorities], key=itemgetter(0))]
        model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

    cutoff = cutoff - (time.time() - ml_start_time)

    # define objective function
    if obj == "cmax":
        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])
    if obj == "twt":
        # total weighted tardiness.
        obj_var = model.NewIntVar(0, 4 * len(all_jobs) * horizon, 'totalweightedtardiness')
        all_tardiness = {}
        for job in all_jobs:
            bound = int(weight[job] * max(horizon - duedate[job], 0))
            tardy = model.NewIntVar(0, bound, 'tardy_%i' % (job))
            model.Add(tardy >=
                      (all_tasks[(job, len(jobs_data[job]) - 1)].end - duedate[job]))  # instance["weight"][job] *
            all_tardiness[job] = tardy
        model.Add(obj_var == sum(weight[job] * all_tardiness[job] for job in all_tardiness))
    if obj == "tmax":
        # Max Tardiness.
        obj_var = model.NewIntVar(0, horizon, 'tmax')
        all_tardiness = {}
        for job in all_jobs:
            bound = int(max(horizon - duedate[job], 0))
            tardy = model.NewIntVar(0, bound, 'tardy_%i' % (job))
            model.Add(tardy >= (all_tasks[(job, len(jobs_data[job]) - 1)].end - duedate[job]))  # instance["weight"][job] *
            all_tardiness[job] = tardy
        model.AddMaxEquality(obj_var, [all_tardiness[job] for job in all_jobs])

    # Solve model.
    solve_start_time = time.time()
    model.Minimize(obj_var)
    solver = cp_model.CpSolver()
    # Sets a time limit of 10 seconds.
    solver.parameters.max_time_in_seconds = cutoff
    status = solver.Solve(model)
    print("CP solving time --- %s seconds ---" % (time.time() - solve_start_time))
    if status == 3:
        print("Infeasible!!!")
        exit()
    print('Number of branches: %i' % solver.NumBranches())
    print('Makespan: %i' % solver.Value(obj_var))

    return solver.Value(obj_var), solver.NumBranches()


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='testing.')
    parser.add_argument('-nmachine', default=10, type=int, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-obj', default="cmax", type=str, help='objective function')  # "cmax", "twt" "tmax"
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance')  # 1.3, 1.5, 1.6
    parser.add_argument('-ml_method', default="DT", type=str, help='ML method')  # "SVM", "DT", "KNN", "MLP", "LR", "NONE"
    parser.add_argument('-level', default="local", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-time_limit', default=60, type=float, help='cutoff time of CP')
    parser.add_argument('-seed', default=10, type=int, help='random seed')
    parser.add_argument('-data_type', default="combine", type=str, help='raw or combine')

    args = parser.parse_args()
    nmachine = args.nmachine
    njob = args.njob
    obj = args.obj
    h = args.duedate
    ml_method = args.ml_method
    time_limit = args.time_limit
    level = args.level
    data_type = args.data_type

    df_raw = pd.DataFrame()
    df_combine = pd.DataFrame()
    df_other = pd.DataFrame()
    allbranches = []
    alltime = []
    allobjs = []
    for seed in range(100, 200):
        print('=========== Seed is %d =========' % seed)
        start_time = time.time()
        objs, branches = solve_jss_ortools(nmachines=nmachine, njobs=njob, seed=args.seed + 99 * seed, obj=obj, h=h,
                                           level=level, ml_model=ml_method, cutoff=time_limit, data_type=data_type)
        print("Totoal running time --- %s seconds ---" % (time.time() - start_time))
        allobjs.append(objs)
        allbranches.append(branches)
        alltime.append(time.time() - start_time)

    allobjs = np.array(allobjs)
    allbranches = np.array(allbranches)
    alltime = np.array(alltime)

    np.savetxt("../results/{}_{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, ml_method, level, nmachine, njob), allobjs, delimiter=",")
    np.savetxt("../results/{}_{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, ml_method, level, nmachine, njob), allbranches, delimiter=",")
    np.savetxt("../results/{}_{}_{}_{}_nm_{}_nj_{}_runtimes.csv".format(obj, h, ml_method, level, nmachine, njob), alltime, delimiter=",")

    print('==================================')
    print("Average running time --- %s seconds ---" % alltime.mean())
    print("Average number of branches --- %s ---" % allbranches.mean())
    print("Average objective values --- %s ---" % allobjs.mean())

