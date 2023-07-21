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


def solve_jss_ortools(nmachines=3, njobs=3, seed=1, obj="tmax", h=1.3, level="global", ml_model="DT", trained_ml="", cutoff=100, data_type="raw"):

    if isinstance(seed, str):
        f = open("../data/" + seed, 'r')
        if seed[:2] != "ta":
            [f.readline() for _ in range(4)]
        size = f.readline()
        N, M = size.split()
        # h = 1.3 # due date allowance
        N = int(N)  # number of jobs
        M = int(M)  # number of machines
        jobs_data = []
        release = {}
        duedate = {}
        weight = {}
        totaljobduration = []
        for i in range(N):
            jobs = f.readline()
            dd = jobs.split()
            nops = M  # rand.randint(1, M) #M #
            jorder = [int(dd[i * 2]) for i in range(nops)]
            jtime = [int(dd[i * 2 + 1]) for i in range(nops)]
            totaldur = sum([jtime[j] for j in range(nops)])
            totaltime = np.sum(jtime)
            release[i] = 0
            duedate[i] = int(release[i] + h * totaltime)
            weight[i] = 4 if i < 0.2 * N else 2 if i < 0.8 * N else 1
            totaljobduration.append(totaldur)
            job = [(jorder[j],
                    jtime[j],
                    sum([jtime[k] for k in range(j)]),
                    sum([jtime[k] for k in range(j, nops)])) for j in range(nops)]
            jobs_data.append(job)
        all_jobs = list(range(N))
        tasklist = []
        dict_task = {}
        loadjob = {}
        loadmachine = {}
        count = 0
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                loadjob[job] = task[1] if job not in loadjob else loadjob[job] + task[1]
                loadmachine[task[0]] = task[1] if task[0] not in loadmachine else loadmachine[task[0]] + task[1]
                dict_task[(job, task_id)] = count
                tasklist.append((task[0], task[1], loadjob[job]))
                count += 1
    else:
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

    # random.seed(seed)
    # """Minimal jobshop problem."""
    #
    # # N = njobs
    # M = nmachines
    # jobs_data = []
    # release = {}
    # duedate = {}
    # weight = {}
    # totaljobduration = []
    # for i in range(N):
    #     nops = M  # random.randint(1, M)
    #     jorder = list(range(M))
    #     random.shuffle(jorder)
    #     jorder = jorder[:nops]
    #     jtime = [random.randint(1,50) for _ in range(nops)]
    #     totaldur = sum([jtime[j] for j in range(nops)])
    #     totaltime = np.sum(jtime)
    #     release[i] = random.randint(0, int(totaltime / 2))
    #     weight[i] = random.choices([1, 2, 4], [0.2, 0.6, 0.2])[0]
    #     duedate[i] = int(release[i] + h * totaltime)
    #     totaljobduration.append(totaldur)
    #     job = [(jorder[j],
    #             jtime[j],
    #             sum([jtime[k] for k in range(j)]),
    #             sum([jtime[k] for k in range(j, nops)])) for j in range(nops)]
    #     jobs_data.append(job)
    # # print(jobs_data)

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
            # release time constraints
            if task_id == 0:
                model.Add(all_tasks[job, task_id].start >= release[job])

    # order decision variables by optimal solution predictions
    ml_start_time = time.time()
    if ml_model != "NONE" and ml_model != "GP":
        # compute features
        disjunctive_raw_features = []
        max_totaljobduration = max(totaljobduration)
        max_duedate = max([duedate[idx] for idx in duedate])
        max_weight = max([weight[w] for w in weight])

        # Create one list of assigned tasks per machine.
        jobs_per_machine = [[] for _ in all_machines]
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                machine = task[0]
                jobs_per_machine[machine].append(assigned_task_type(start=0, job=job, index=task_id))

        for machine in all_machines:
            operation_raw_features = []
            for assigned_task in jobs_per_machine[machine]:
                if ml_model == "GPc" or ml_model == "DT":
                    operation_raw_features.append([
                        float(assigned_task.index),
                        float(jobs_data[assigned_task.job][assigned_task.index][1]),
                        float(jobs_data[assigned_task.job][assigned_task.index][2]),
                        float(jobs_data[assigned_task.job][assigned_task.index][3]),
                        float(totaljobduration[assigned_task.job]),
                        float(duedate[assigned_task.job]),
                        float(weight[assigned_task.job]),
                        float(release[assigned_task.job]),
                        float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start][
                                  "earliest_start"]),
                        float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start]["workload"])
                    ])
                else:
                    operation_raw_features.append([
                        float(assigned_task.index) / nops,
                        float(jobs_data[assigned_task.job][assigned_task.index][1]) / max_totaljobduration,
                        float(jobs_data[assigned_task.job][assigned_task.index][2]) / max_totaljobduration,
                        float(jobs_data[assigned_task.job][assigned_task.index][3]) / max_totaljobduration,
                        float(totaljobduration[assigned_task.job]) / max_totaljobduration,
                        float(duedate[assigned_task.job]) / max_totaljobduration,
                        float(weight[assigned_task.job]) / max_weight,
                        float(release[assigned_task.job]) / max_totaljobduration,
                        float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start][
                                  "earliest_start"]) / max_totaljobduration,
                        float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start]["workload"]) /
                        feature_dict[all_tasks[assigned_task.job, assigned_task.index].start]["maxload"]
                    ])

            for i in range(len(operation_raw_features)):
                for j in range(len(operation_raw_features)):
                    if i < j:
                        disjunctive_raw_features.append([
                            operation_raw_features[i][0] - operation_raw_features[j][0],
                            operation_raw_features[i][1] - operation_raw_features[j][1],
                            operation_raw_features[i][2] - operation_raw_features[j][2],
                            operation_raw_features[i][3] - operation_raw_features[j][3],
                            operation_raw_features[i][4] - operation_raw_features[j][4],
                            operation_raw_features[i][5] - operation_raw_features[j][5],
                            operation_raw_features[i][6] - operation_raw_features[j][6],
                            operation_raw_features[i][7] - operation_raw_features[j][7],
                            operation_raw_features[i][8] - operation_raw_features[j][8],
                            # operation_raw_features[i][9] - operation_raw_features[j][9],
                        ])

        # compute class labels for each pair of operations
        # predict class labels
        predicted_label = trained_ml.predict(np.array(disjunctive_raw_features))

        # compute the priorties of decision variables based on ML prediction
        var_priorities_ml = {}
        for var in list_of_vars:
            (job, task_id) = var[2]
            var_priorities_ml[all_tasks[job, task_id].start] = 0

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

        # sort variables in order based on the priorties
        var_select_order = [x for _, x in sorted([(-var_priorities_ml[v], v) for v in var_priorities_ml],
                                                 key=itemgetter(0))]
        model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)
    elif ml_model == "GP":
        priorities = []
        varlist = []
        for var in feature_dict:
            # print([c for c in feature_values[var]])
            feature_inputs = [
                feature_dict[var]["earliest_start"],
                feature_dict[var]["duration"],
                feature_dict[var]["weight"],
                feature_dict[var]["duedate"],
                feature_dict[var]["release"],
                feature_dict[var]["total_dur"],
                feature_dict[var]["workload"],
                feature_dict[var]["maxload"]
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
            priorities.append(out)
            varlist.append(var)
        # var_select_order = [x for _, x in sorted(zip(priorities, varlist))]
        var_select_order = [x for _, x in sorted(zip(priorities, varlist), key=itemgetter(0))]
        model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

    cutoff = cutoff - (time.time() - ml_start_time)




    # Solve model.
    solve_start_time = time.time()
    # model.Minimize(obj_var)
    solver = cp_model.CpSolver()
    # Sets a time limit of 10 seconds.
    solver.parameters.max_time_in_seconds = cutoff
    status = solver.Solve(model)
    print("CP solving time --- %s seconds ---" % (time.time() - solve_start_time))
    if status == 3:
        print("Infeasible!!!")
        exit()
    print('Number of branches: %i' % solver.NumBranches())
    # print('objVal: %i' % solver.Value(obj_var))

    # define objective function
    if obj == "cmax":
        # Makespan objective.
        # obj_var = model.NewIntVar(0, horizon, 'makespan')
        # model.AddMaxEquality(obj_var, [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])
        obj_val = max([solver.Value(all_tasks[(job, len(jobs_data[job]) - 1)].end) for job in all_jobs])
    if obj == "twt":

        # add tardiness constraints.
        all_tardiness = {}
        for job in all_jobs:
            tardy = solver.Value(all_tasks[(job, len(jobs_data[job]) - 1)].end) - duedate[job]
            if tardy < 0:
                all_tardiness[job] = 0
            else:
                all_tardiness[job] = tardy
        obj_val = sum(weight[job] * all_tardiness[job] for job in all_jobs)

        # model.Add(obj_var == sum(weight[job] * all_tardiness[job] for job in all_tardiness))
    if obj == "tmax":
        # add tardiness constraints.
        all_tardiness = {}
        for job in all_jobs:
            tardy = solver.Value(all_tasks[(job, len(jobs_data[job]) - 1)].end) - duedate[job]
            if tardy < 0:
                all_tardiness[job] = 0
            else:
                all_tardiness[job] = tardy
        obj_val = max(weight[job] * all_tardiness[job] for job in all_jobs)

    return obj_val, solver.NumBranches()


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='testing.')
    parser.add_argument('-nmachine', default=10, type=int, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-idx', default=0, type=int, help='the index of dataset')
    parser.add_argument('-obj', default="cmax", type=str, help='objective function')  # "cmax", "twt" "tmax"
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance')  # 1.3, 1.5, 1.6
    parser.add_argument('-ml_method', default="DT", type=str, help='ML method')  # "SVM", "DT", "KNN", "MLP", "LR", "NONE"
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

    # files = ['la01', 'la07']
    trained_ml = []
    if ml_method != "NONE" and ml_method != "GP":
        trained_ml = joblib.load("../training_data/{}_{}_{}_{}.pkl".format(obj, ml_method, level, data_type))


    if ml_method == "GPc":
        print(trained_ml._program)

    if ml_method == "LR" or ml_method == "SVM":
        coefs = trained_ml.coef_[0]
        print([float("{:.3f}".format(v)) for v in coefs])

        inter = trained_ml.intercept_[0]
        print(inter)
        # print(float("{:.3f}".format(inter)))

        #     print("{:.3f}".format(v))
        #
        # print(["{:.3f}".format(v) for v in ])

    df_raw = pd.DataFrame()
    df_combine = pd.DataFrame()
    df_other = pd.DataFrame()
    allbranches = []
    alltime = []
    allobjs = []
    for seed in range(100, 200):
    # for seed in range(100, 101):
    # for seed in files:
        # seed = files[idx]
        print('=========== Seed is {} ========='.format(seed))
        start_time = time.time()
        if isinstance(seed, str):
            objs, branches = solve_jss_ortools(nmachines=nmachine, njobs=njob, seed=seed, obj=obj, h=h,
                                                level=level, ml_model=ml_method, trained_ml=trained_ml, cutoff=time_limit, data_type=data_type)
        else:
            objs, branches = solve_jss_ortools(nmachines=nmachine, njobs=njob, seed=args.seed + 99 * seed, obj=obj, h=h,
                                               level=level, ml_model=ml_method, trained_ml=trained_ml, cutoff=time_limit, data_type=data_type)


        print("Totoal running time --- %s seconds ---" % (time.time() - start_time))
        allobjs.append(objs)
        allbranches.append(branches)
        alltime.append(time.time() - start_time)

    allobjs = np.array(allobjs)
    allbranches = np.array(allbranches)
    alltime = np.array(alltime)

    np.savetxt("../results_ini/{}_{}_{}_{}_{}_nm_{}_nj_{}_objs.csv".format(obj, h, ml_method, level, data_type, nmachine, njob), allobjs, delimiter=",")
    np.savetxt("../results_ini/{}_{}_{}_{}_{}_nm_{}_nj_{}_branches.csv".format(obj, h, ml_method, level, data_type, nmachine, njob), allbranches, delimiter=",")
    np.savetxt("../results_ini/{}_{}_{}_{}_{}_nm_{}_nj_{}_runtimes.csv".format(obj, h, ml_method, level, data_type, nmachine, njob), alltime, delimiter=",")


    # np.savetxt("../results/benchmark_{}_{}_{}_objs.csv".format(obj, ml_method, data_type), allobjs, delimiter=",")
    # np.savetxt("../results/benchmark_{}_{}_{}_branches.csv".format(obj, ml_method, data_type), allbranches, delimiter=",")
    # np.savetxt("../results/benchmark_{}_{}_{}_runtimes.csv".format(obj, ml_method, data_type), alltime, delimiter=",")


    print('==================================')
    print("Average running time --- %s seconds ---" % alltime.mean())
    print("Average number of branches --- %s ---" % allbranches.mean())
    print("Average objective values --- %s ---" % allobjs.mean())

