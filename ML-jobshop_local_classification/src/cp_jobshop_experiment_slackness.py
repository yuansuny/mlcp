from __future__ import print_function
import collections
from ortools.sat.python import cp_model
import numpy as np
import random
import argparse
import pandas as pd
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

def solve_jss_ortools(nmachines=3, njobs=3, seed=1, obj="tmax", h=1.3, level="global", ml_model="DT", cutoff=100):
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
        jtime = [random.randint(1, 50) for _ in range(nops)]
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
    assigned_task_type = collections.namedtuple('assigned_task_type','start job index')

    objs = []
    branches = []
    runtimes = []



    # solve the original problem instance
    start_time = time.time()
    # Create the model.
    model = cp_model.CpModel()
    # Create jobs.
    all_tasks = {}
    list_of_vars = []
    count = 0
    for job in all_jobs:
        for task_id, task in enumerate(jobs_data[job]):
            start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, task_id))
            duration = task[1]
            end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
            all_tasks[job, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            list_of_vars.append((all_tasks[job, task_id].end, task[0], (job, task_id), count))
            count += 1

    # Create and add disjunctive constraints.
    for machine in all_machines:
        intervals = []
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                if task[0] == machine:
                    intervals.append(all_tasks[job, task_id].interval)
        model.AddNoOverlap(intervals)

    # Add precedence contraints.
    for job in all_jobs:
        for task_id in range(0, len(jobs_data[job]) - 1):
            model.Add(all_tasks[job, task_id + 1].start >= all_tasks[job, task_id].end)

    # define objective function
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])

    # Solve model.
    model.Minimize(obj_var)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == 3:
        print("Infeasible!!!")
        exit()
    # print('Number of branches: %i' % solver.NumBranches())
    print('Makespan: %i' % solver.Value(obj_var))
    objs.append(solver.Value(obj_var))
    branches.append(solver.NumBranches())
    runtimes.append(time.time() - start_time)

    optimal_obj = solver.Value(obj_var)

    var_starting = []
    for var in list_of_vars:
        (job, task_id) = var[2]
        var_starting.append(solver.Value(all_tasks[job, task_id].start))
    # print(var_starting)


    # var_slackness = []
    # for var in list_of_vars:
    #     var_slackness.append(solver.Value(var[0]))
    # print(var_slackness)




    # resolve the model to push all operations as early as possible
    start_time = time.time()
    # Create the model.
    model = cp_model.CpModel()
    # Create jobs.
    all_tasks = {}
    list_of_vars = []
    count = 0
    for job in all_jobs:
        for task_id, task in enumerate(jobs_data[job]):
            start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, task_id))
            duration = task[1]
            end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
            all_tasks[job, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            list_of_vars.append((all_tasks[job, task_id].end, task[0], (job, task_id), count))
            count += 1

    # Create and add disjunctive constraints.
    for machine in all_machines:
        intervals = []
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                if task[0] == machine:
                    intervals.append(all_tasks[job, task_id].interval)
        model.AddNoOverlap(intervals)

    # Add precedence constraints.
    for job in all_jobs:
        for task_id in range(0, len(jobs_data[job]) - 1):
            model.Add(all_tasks[job, task_id + 1].start >= all_tasks[job, task_id].end)

    var_priorities = {}
    for var in list_of_vars:
        (job, task_id) = var[2]
        idx = var[3]
        var_priorities[all_tasks[job, task_id].start] = var_starting[idx]

    var_select_order = [x for _, x in sorted([(var_priorities[v], v) for v in var_priorities], key=itemgetter(0))]
    model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

    # define makespan constraint
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])
    model.Add(obj_var <= optimal_obj)

    # define another objective on slackness
    slack_var = model.NewIntVar(0, M * len(all_jobs) * horizon, 'slackness')
    model.Add(slack_var >= sum(var[0] for var in list_of_vars))

    # Solve model.
    model.Minimize(slack_var)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == 3:
        print("Infeasible!!!")
        exit()
    # print('Number of branches: %i' % solver.NumBranches())
    print('Makespan: %i' % solver.Value(obj_var))
    objs.append(solver.Value(obj_var))
    branches.append(solver.NumBranches())
    runtimes.append(time.time() - start_time)

    var_slackness_early = []
    for var in list_of_vars:
        var_slackness_early.append(solver.Value(var[0]))
    # print(var_slackness_early)

    var_starting = []
    for var in list_of_vars:
        (job, task_id) = var[2]
        var_starting.append(solver.Value(all_tasks[job, task_id].start))
    # print(var_starting)





    # resolve the model to push all operations as late as possible
    start_time = time.time()
    # Create the model.
    model = cp_model.CpModel()
    # Create jobs.
    all_tasks = {}
    list_of_vars = []
    count = 0
    for job in all_jobs:
        for task_id, task in enumerate(jobs_data[job]):
            start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, task_id))
            duration = task[1]
            end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
            all_tasks[job, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            list_of_vars.append((all_tasks[job, task_id].end, task[0], (job, task_id), count))
            count += 1

    # Create and add disjunctive constraints.
    for machine in all_machines:
        intervals = []
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                if task[0] == machine:
                    intervals.append(all_tasks[job, task_id].interval)
        model.AddNoOverlap(intervals)

    # Add precedence constraints.
    for job in all_jobs:
        for task_id in range(0, len(jobs_data[job]) - 1):
            model.Add(all_tasks[job, task_id + 1].start >= all_tasks[job, task_id].end)

    var_priorities = {}
    for var in list_of_vars:
        (job, task_id) = var[2]
        idx = var[3]
        var_priorities[all_tasks[job, task_id].start] = var_starting[idx]

    var_select_order = [x for _, x in sorted([(var_priorities[v], v) for v in var_priorities], key=itemgetter(0))]
    model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

    # define makespan constraint
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])
    model.Add(obj_var <= optimal_obj)

    # define another objective on slackness
    slack_var = model.NewIntVar(0, M * len(all_jobs) * horizon, 'slackness')
    model.Add(slack_var <= sum(var[0] for var in list_of_vars))

    # Solve model.
    model.Maximize(slack_var)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == 3:
        print("Infeasible!!!")
        exit()
    # print('Number of branches: %i' % solver.NumBranches())
    print('Makespan: %i' % solver.Value(obj_var))
    objs.append(solver.Value(obj_var))
    branches.append(solver.NumBranches())
    runtimes.append(time.time() - start_time)

    var_slackness_late = []
    end_var_time = []
    for var in list_of_vars:
        var_slackness_late.append(solver.Value(var[0]))
    # print(var_slackness_late)




    # resolve the model based on the ordering of slackness and optimal solution
    start_time = time.time()
    # Create the model.
    model = cp_model.CpModel()
    # Create jobs.
    all_tasks = {}
    list_of_vars = []
    count = 0
    for job in all_jobs:
        for task_id, task in enumerate(jobs_data[job]):
            start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, task_id))
            duration = task[1]
            end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
            all_tasks[job, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            list_of_vars.append((all_tasks[job, task_id].end, task[0], (job, task_id), count))
            count += 1

    # Create and add disjunctive constraints.
    for machine in all_machines:
        intervals = []
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                if task[0] == machine:
                    intervals.append(all_tasks[job, task_id].interval)
        model.AddNoOverlap(intervals)

    # Add precedence constraints.
    for job in all_jobs:
        for task_id in range(0, len(jobs_data[job]) - 1):
            model.Add(all_tasks[job, task_id + 1].start >= all_tasks[job, task_id].end)

    # ordering decision variables
    var_priorities = {}
    for var in list_of_vars:
        (job, task_id) = var[2]
        idx = var[3]
        # var_priorities[all_tasks[job, task_id].start] = var_slackness_late[idx] - var_slackness_early[idx]
        # var_priorities[all_tasks[job, task_id].start] = var_slackness_late[idx] - var_slackness_early[idx] + \
        #                                                 float(var_slackness_late[idx]) / horizon
        # var_priorities[all_tasks[job, task_id].start] = var_slackness_late[idx] - var_slackness_early[idx] + \
        #                                                 float(var_starting[idx]) / horizon
        var_priorities[all_tasks[job, task_id].start] = var_starting[idx] + float(var_slackness_late[idx]-var_slackness_early[idx])/horizon

    var_select_order = [x for _, x in sorted([(var_priorities[v], v) for v in var_priorities], key=itemgetter(0))]
    model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)


    # define makespan objective
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])

    # Solve model.
    model.Minimize(obj_var)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == 3:
        print("Infeasible!!!")
        exit()
    # print('Number of branches: %i' % solver.NumBranches())
    print('Makespan: %i' % solver.Value(obj_var))
    objs.append(solver.Value(obj_var))
    branches.append(solver.NumBranches())
    runtimes.append(time.time() - start_time)

    return objs, branches, runtimes



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='JSS solver.')
    parser.add_argument('-nmachine', default=10, type=int, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-obj', default="cmax", type=str, help='objective function')  # "cmax", "twt" "tmax"
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance')  # 1.3, 1.5, 1.6
    parser.add_argument('-ml_method', default="DT", type=str, help='ML method')  # "SVM", "DT", "KNN", "MLP", "NONE"
    parser.add_argument('-level', default="global", type=str, help='ordering level')  # "global", "local"
    parser.add_argument('-time_limit', default=60, type=float, help='cutoff time of CP')
    parser.add_argument('-seed', default=10, type=int, help='random seed')

    args = parser.parse_args()
    nmachine = args.nmachine
    njob = args.njob
    obj = args.obj
    h = args.duedate
    ml_method = args.ml_method
    time_limit = args.time_limit
    level = args.level

    df_raw = pd.DataFrame()
    df_combine = pd.DataFrame()
    df_other = pd.DataFrame()
    import matplotlib.pyplot as plt
    import numpy as np

    ninstances = 100
    # for mode in ["add constraints", "order"]:
    for mode in ["order"]:
        allobjs = []
        allbranches = []
        allruntimes = []
        # for seed in range(ninstances):
        for seed in range(100, 200):
            print('=========== Seed is %d =========' % seed)
            objs, branches, runtimes = solve_jss_ortools(nmachines=nmachine, njobs=njob, seed=args.seed + 99 * seed,
                                                         obj=obj, h=h, level=level, ml_model=ml_method,
                                                         cutoff=time_limit)
            print(objs, branches, runtimes)
            allobjs.append(objs)
            allbranches.append(branches)
            allruntimes.append(runtimes)

        allobjs = np.array(allobjs)
        allbranches = np.array(allbranches)
        allruntimes = np.array(allruntimes)
        # np.savetxt("../results/{}_slackness_nm_{}_nj_{}_branches.csv".format(obj, nmachine, njob), allbranches, delimiter=",")
        # np.savetxt("../results/{}_slackness_nm_{}_nj_{}_runtimes.csv".format(obj, nmachine, njob), allruntimes, delimiter=",")
        # np.savetxt("../results/{}_slackness_nm_{}_nj_{}_objs.csv".format(obj, nmachine, njob), allobjs, delimiter=",")

        # plt.plot(np.arange(ninstances), allbranches[:, 0], label="default")
        # plt.plot(np.arange(ninstances), allbranches[:, 1], label="resolve")

        # plt.figure()
        # plt.plot(range(len(allobjs)), allbranches[:, 0], label="default")
        # plt.plot(range(len(allobjs)), allbranches[:, 1], label="resolve")
        # plt.legend()
        # plt.title("Resolve mode = {}".format(mode))
        # plt.show()
        #
        # plt.figure()
        # plt.plot(range(len(allobjs)), allruntimes[:, 0], label="default")
        # plt.plot(range(len(allobjs)), allruntimes[:, 1], label="resolve")
        # plt.legend()
        # plt.title("Resolve mode = {}".format(mode))
        # plt.show()

        print('==============runtime=================')
        print("Average running time of Default--- %s seconds ---" % allruntimes[:, 0].mean())
        print("Average running time of push er--- %s seconds ---" % allruntimes[:, 1].mean())
        print("Average running time of push la--- %s seconds ---" % allruntimes[:, 2].mean())
        print("Average running time of slackne--- %s seconds ---" % allruntimes[:, 3].mean())

        print('==============branches=================')
        print("Average number of branches of Default--- %s seconds ---" % allbranches[:, 0].mean())
        print("Average number of branches of push er--- %s seconds ---" % allbranches[:, 1].mean())
        print("Average number of branches of push la--- %s seconds ---" % allbranches[:, 2].mean())
        print("Average number of branches of slackne--- %s seconds ---" % allbranches[:, 3].mean())


