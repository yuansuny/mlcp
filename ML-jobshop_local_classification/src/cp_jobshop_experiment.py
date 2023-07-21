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
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index')

    resolve = False
    new_constraints = []
    objs = []
    branches = []
    runtimes = []
    while True:
        start_time = time.time()
        # Create the model.
        model = cp_model.CpModel()
        # Create jobs.
        all_tasks = {}
        list_of_vars = []
        count = 0
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                start_var = model.NewIntVar(0, horizon,
                                            'start_%i_%i' % (job, task_id))
                duration = task[1]
                end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
                interval_var = model.NewIntervalVar(
                    start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
                all_tasks[job, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var)
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
                model.Add(all_tasks[job, task_id +
                                    1].start >= all_tasks[job, task_id].end)

        var_select_order = []
        var_priorities = {}
        if len(new_constraints) > 0:
            for pair in new_constraints:
                if pair[0] not in var_priorities:
                    var_priorities[pair[0]] = 1
                else:
                    var_priorities[pair[0]] += 1
            var_select_order = [x for _, x in sorted([(-var_priorities[v], v) for v in var_priorities], key=itemgetter(0))]
            model.AddDecisionStrategy(var_select_order, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

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
                          (all_tasks[(job, len(jobs_data[job]) - 1)].end - duedate[
                              job]))  # instance["weight"][job] *
                all_tardiness[job] = tardy
            model.Add(obj_var == sum(weight[job] * all_tardiness[job] for job in all_tardiness))
        if obj == "tmax":
            # Max Tardiness.
            obj_var = model.NewIntVar(0, horizon, 'tmax')
            all_tardiness = {}
            for job in all_jobs:
                bound = int(max(horizon - duedate[job], 0))
                tardy = model.NewIntVar(0, bound, 'tardy_%i' % (job))
                model.Add(tardy >= (all_tasks[(job, len(jobs_data[job]) - 1)].end - duedate[
                    job]))  # instance["weight"][job] *
                all_tardiness[job] = tardy
            model.AddMaxEquality(obj_var, [all_tardiness[job] for job in all_jobs])

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

        if status == cp_model.OPTIMAL:
            disjunctive_raw_features = []
            disjunctive_combine_features = []
            labels = []
            # Print out makespan.
            # print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
            # print()

            # Create one list of assigned tasks per machine.
            assigned_jobs = [[] for _ in all_machines]
            mops = {i: [] for i in range(nmachines)}

            for job in all_jobs:
                for task_id, task in enumerate(jobs_data[job]):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(all_tasks[job, task_id].start),
                            job=job,
                            index=task_id))
                    mops[machine].append(all_tasks[job, task_id].start)

            for m in range(nmachines):
                for i in range(len(mops[m])):
                    for j in range(len(mops[m])):
                        if i > j:
                            op1 = mops[m][i]
                            op2 = mops[m][j]
                            if solver.Value(op1) < solver.Value(op2):
                                new_constraints.append((op1, op2))
                            else:
                                new_constraints.append((op2, op1))


            optimal_obj = int(solver.ObjectiveValue())


            # print(sol_line_tasks)
            # print('Task Time Intervals\n')
            # print(sol_line)

            if resolve:
                break
            else:
                resolve = True
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

    allobjs = []
    allbranches = []
    allruntimes = []
    # for seed in range(ninstances):
    for seed in range(100, 200):
        print('=========== Seed is %d =========' % seed)
        objs, branches, runtimes = solve_jss_ortools(nmachines=nmachine, njobs=njob, seed=args.seed + 99 * seed,
                                           obj=obj, h=h, level=level, ml_model=ml_method, cutoff=time_limit)
        print(objs, branches, runtimes)
        allobjs.append(objs)
        allbranches.append(branches)
        allruntimes.append(runtimes)

    allobjs = np.array(allobjs)
    allbranches = np.array(allbranches)
    allruntimes = np.array(allruntimes)
    np.savetxt("../results/{}_optimal_local_nm_{}_nj_{}_branches.csv".format(obj,nmachine, njob), allbranches, delimiter=",")
    np.savetxt("../results/{}_optimal_local_nm_{}_nj_{}_runtimes.csv".format(obj,nmachine, njob), allruntimes, delimiter=",")
    np.savetxt("../results/{}_optimal_local_nm_{}_nj_{}_objs.csv".format(obj,nmachine, njob), allobjs, delimiter=",")

    # plt.plot(np.arange(ninstances), allbranches[:, 0], label="default")
    # plt.plot(np.arange(ninstances), allbranches[:, 1], label="resolve")

    # plt.figure()
    # plt.plot(range(1, 21), allbranches[:, 0], label="default")
    # plt.plot(range(1, 21), allbranches[:, 1], label="resolve")
    # plt.legend()
    # plt.title("Resolve mode = {}".format(mode))
    # plt.show()
    #
    # plt.figure()
    # plt.plot(range(1, 21), allruntimes[:, 0], label="default")
    # plt.plot(range(1, 21), allruntimes[:, 1], label="resolve")
    # plt.legend()
    # plt.title("Resolve mode = {}".format(mode))
    # plt.show()

    print('===============Default=================')
    print("Average running time of Default--- %s seconds ---" % allruntimes[:, 0].mean())
    print("Average number of branches of Default--- %s ---" % allbranches[:, 0].mean())
    print("Average objective value of Default--- %s ---" % allobjs[:, 0].mean())

    print('================ML Ordering==================')
    print("Average running time of ML--- %s seconds ---" % allruntimes[:, 1].mean())
    print("Average number of branches of ML--- %s ---" % allbranches[:, 1].mean())
    print("Average objective value Default--- %s ---" % allobjs[:, 1].mean())

