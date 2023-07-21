from __future__ import print_function
import collections
from ortools.sat.python import cp_model
import numpy as np
import random
import argparse
import pandas as pd


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

def solve_jss_ortools(nmachines=3, njobs=3, h=1.6, objective="cmax", seed=1):
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
        nops = M # random.randint(1, M)
        jorder = list(range(M))
        random.shuffle(jorder)
        jorder = jorder[:nops]
        jtime = [random.randint(1,50) for _ in range(nops)]
        totaldur = sum([jtime[j] for j in range(nops)])
        totaltime = np.sum(jtime)
        release[i] = random.randint(0, int(totaltime/2))
        duedate[i] = int(release[i] + h*totaltime)
        weight[i] = random.choices([1, 2, 4], [0.2, 0.6, 0.2])[0]
        totaljobduration.append(totaldur)
        job = [(jorder[j],
                jtime[j],
                sum([jtime[k] for k in range(j)]),
                sum([jtime[k] for k in range(j, nops)])) for j in range(nops)]
        jobs_data.append(job)
    print(jobs_data)

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
            feature_dict[
                start_var] = {}  # [task_id, earliest_start, duration, 0, 0,  0]  # task_id, earliest_start, duration, workload, sum_early
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

    # optimal_obj = None
    # # Create the model.
    # model = cp_model.CpModel()
    # # Create jobs.
    # all_tasks = {}
    # list_of_vars = []
    # feature_dict = {}
    # count = 0
    # for job in all_jobs:
    #     for task_id, task in enumerate(jobs_data[job]):
    #         start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, task_id))
    #         feature_dict[start_var] = {}
    #         duration = task[1]
    #         end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, task_id))
    #         interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval_%i_%i' % (job, task_id))
    #         all_tasks[job, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
    #         list_of_vars.append((all_tasks[job, task_id].end, task[0], (job, task_id), count))
    #         count += 1




    # # Create and add disjunctive constraints.
    # for machine in all_machines:
    #     intervals = []
    #     for job in all_jobs:
    #         for task_id, task in enumerate(jobs_data[job]):
    #             if task[0] == machine:
    #                 intervals.append(all_tasks[job, task_id].interval)
    #     model.AddNoOverlap(intervals)

    # Create and add disjunctive constraints.
    maxload = 0
    for machine in all_machines:
        intervals = []
        matched_varlist = []
        workload = 0
        # sum_early_start = 0
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                if task[0] == machine:
                    intervals.append(all_tasks[job, task_id].interval)
                    matched_varlist.append(all_tasks[job, task_id].start)
                    workload += task[1]
                    # sum_early_start += feature_dict[all_tasks[job, task_id].start]["earliest_start"]
        model.AddNoOverlap(intervals)
        for var in matched_varlist:
            feature_dict[var]["workload"] = workload
            maxload = max(workload, maxload)
    for var in feature_dict:
        feature_dict[var]["maxload"] = maxload

    # Add precedence contraints.
    for job in all_jobs:
        for task_id in range(0, len(jobs_data[job]) - 1):
            model.Add(all_tasks[job, task_id + 1].start >= all_tasks[job, task_id].end)
            # release time constraints
            if task_id == 0:
                model.Add(all_tasks[job, task_id].start >= release[job])

    # define objective function
    if objective == "cmax":
        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])
    if objective == "twt":
        # total weighted tardiness.
        obj_var = model.NewIntVar(0, 4*len(all_jobs)*horizon, 'totalweightedtardiness')
        all_tardiness = {}
        for job in all_jobs:
            bound = int(weight[job]*max(horizon-duedate[job], 0))
            tardy = model.NewIntVar(0, bound, 'tardy_%i' % (job))
            model.Add(tardy >= (all_tasks[(job, len(jobs_data[job]) - 1)].end - duedate[job])) #instance["weight"][job] *
            all_tardiness[job] = tardy
        model.Add(obj_var == sum(weight[job] * all_tardiness[job] for job in all_jobs))
    if objective == "tmax":
        # Max Tardiness.
        obj_var = model.NewIntVar(0, horizon, 'tmax')
        all_tardiness = {}
        for job in all_jobs:
            bound = int(max(horizon-duedate[job], 0))
            tardy = model.NewIntVar(0, bound, 'tardy_%i' % (job))
            model.Add(tardy >= (all_tasks[(job, len(jobs_data[job]) - 1)].end - duedate[job])) #instance["weight"][job] *
            all_tardiness[job] = tardy
        model.AddMaxEquality(obj_var, [all_tardiness[job] for job in all_jobs])

    # Solve model.
    model.Minimize(obj_var)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == 3:
        print("Infeasible!!!")
        exit()
    print('Number of branches: %i' % solver.NumBranches())
    print('Objective value: %i' % solver.Value(obj_var))

    if status == cp_model.OPTIMAL:
        disjunctive_raw_features = []
        labels = []
        # Print out the optimal objective value.
        optimal_obj = int(solver.ObjectiveValue())
        print('Optimal objective value: %i' % solver.ObjectiveValue())

        # compute the maximum ending time for normalization
        max_end = max([solver.Value(var[0]) for var in list_of_vars])

        # construct training data
        max_totaljobduration = max(totaljobduration)
        max_duedate = max([duedate[idx] for idx in duedate])
        max_weight = max([weight[w] for w in weight])

        # Create one list of assigned tasks per machine.
        assigned_jobs = [[] for _ in all_machines]
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job, task_id].start),
                        job=job,
                        index=task_id))


        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            operation_raw_features = []
            for assigned_task in assigned_jobs[machine]:
                operation_raw_features.append([
                    # float(assigned_task.index),
                    # float(jobs_data[assigned_task.job][assigned_task.index][1]),
                    # float(jobs_data[assigned_task.job][assigned_task.index][2]),
                    # float(jobs_data[assigned_task.job][assigned_task.index][3]),
                    # float(totaljobduration[assigned_task.job]),
                    # float(duedate[assigned_task.job]),
                    # float(weight[assigned_task.job]),
                    # float(release[assigned_task.job]),
                    # float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start][
                    #           "earliest_start"]),
                    # float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start]["workload"]),
                    float(assigned_task.index) / nops,
                    float(jobs_data[assigned_task.job][assigned_task.index][1]) / max_totaljobduration,
                    float(jobs_data[assigned_task.job][assigned_task.index][2]) / max_totaljobduration,
                    float(jobs_data[assigned_task.job][assigned_task.index][3]) / max_totaljobduration,
                    float(totaljobduration[assigned_task.job]) / max_totaljobduration,
                    float(duedate[assigned_task.job]) / max_totaljobduration,
                    float(weight[assigned_task.job]) / max_weight,
                    float(release[assigned_task.job]) / max_totaljobduration,
                    float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start]["earliest_start"]) / max_totaljobduration,
                    float(feature_dict[all_tasks[assigned_task.job, assigned_task.index].start]["workload"]) /
                    feature_dict[all_tasks[assigned_task.job, assigned_task.index].start]["maxload"]
                ])


            for i in range(len(operation_raw_features)):
                for j in range(len(operation_raw_features)):
                    if j != i:
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
                            operation_raw_features[i][9] - operation_raw_features[j][9],
                        ])
                        labels.append(1 if i < j else 0)

        return disjunctive_raw_features, labels, optimal_obj, seed


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='JSS solver.')
    parser.add_argument('-nmachine', default=10, type=int, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-duedate', default=1.3, type=float, help='due date allowance') # 1.3, 1.5, 1.6
    parser.add_argument('-seed', default=10, type=int, help='random seed')
    parser.add_argument('-obj', default="tmax", type=str, help='objective functions') # "cmax", "twt" "tmax"
    args = parser.parse_args()
    nmachine = args.nmachine
    njob = args.njob
    obj = args.obj
    h = args.duedate
    df_raw = pd.DataFrame()
    # for seed in range(1000, 5000):
    # for seed in range(1000, 1500):
    for seed in range(0, 100):
        disjunctive_raw_features, labels, optobj, _seed = solve_jss_ortools(nmachines=nmachine, njobs=njob, h=h, objective=obj, seed=args.seed + 10*seed)
        size = len(labels)
        data_raw = np.hstack((np.array(disjunctive_raw_features).reshape(size, -1), np.array(labels).reshape(size, -1)))
        df_raw = df_raw.append(pd.DataFrame(data_raw))
        print("Complete solving instance #{}".format(_seed))

    df_raw.columns = ["opid", "duration", "upstreamdur", "downstreamdur", "totaljobdur", "duedate", "weight", "release", "earliest_start", "workload", "label"]
    df_raw.to_csv("../training_data_norm/training_data_{}_classification_nm_{}_nj_{}_h{}.csv".format(obj, nmachine, njob, h), index=False)

