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

def solve_jss_ortools(nmachines=3, njobs=3, seed=1):
    random.seed(seed)
    """Minimal jobshop problem."""

    N = njobs
    M = nmachines
    jobs_data = []
    totaljobduration = []
    for i in range(N):
        nops = M # random.randint(1, M)
        jorder = list(range(M))
        random.shuffle(jorder)
        jorder = jorder[:nops]
        jtime = [random.randint(1,50) for _ in range(nops)]
        totaldur = sum([jtime[j] for j in range(nops)])
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
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index')

    optimal_obj = None
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

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(
        obj_var,
        [all_tasks[(job, len(jobs_data[job]) - 1)].end for job in all_jobs])
    # model.Add(obj_var <= 360)


    # Solve model.
    model.Minimize(obj_var)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == 3:
        print("Infeasible!!!")
        exit()
    print('Number of branches: %i' % solver.NumBranches())
    print('Makespan: %i' % solver.Value(obj_var))


    if status == cp_model.OPTIMAL:
        disjunctive_raw_features = []
        disjunctive_combine_features = []
        labels = []
        # Print out makespan.
        print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
        print()

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

        disp_col_width = 10
        sol_line = ''
        sol_line_tasks = ''
        optimal_obj = int(solver.ObjectiveValue())

        print('Optimal Schedule', '\n')

        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line += 'Machine ' + str(machine) + ': '
            sol_line_tasks += 'Machine ' + str(machine) + ': '

            operation_raw_features = []
            workload = 0
            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += name + ' ' * (disp_col_width - len(name))
                start = assigned_task.start
                duration = jobs_data[assigned_task.job][assigned_task.index][1]
                workload += duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += sol_tmp + ' ' * (disp_col_width - len(sol_tmp))
                operation_raw_features.append([assigned_task.index, # operation index
                                               jobs_data[assigned_task.job][assigned_task.index][1], # operation duration
                                               jobs_data[assigned_task.job][assigned_task.index][2], # earliest start time (total upstream durations)
                                               jobs_data[assigned_task.job][assigned_task.index][3], # earliest completion time
                                               totaljobduration[assigned_task.job] # total duration of job the operation belong to
                                               ])

            for i in range(len(operation_raw_features)):
                for j in range(len(operation_raw_features)):
                    if j !=i:
                        disjunctive_raw_features.append([
                            operation_raw_features[i][0], operation_raw_features[j][0],
                            operation_raw_features[i][1], operation_raw_features[j][1],
                            operation_raw_features[i][2], operation_raw_features[j][2],
                            operation_raw_features[i][3], operation_raw_features[j][3],
                            operation_raw_features[i][4], operation_raw_features[j][4],
                        ])

                        disjunctive_combine_features.append([
                            operation_raw_features[i][0] - operation_raw_features[j][0],
                            operation_raw_features[i][1] - operation_raw_features[j][1],
                            operation_raw_features[i][2] - operation_raw_features[j][2],
                            operation_raw_features[i][3] - operation_raw_features[j][3],
                            operation_raw_features[i][4] - operation_raw_features[j][4],
                        ])

                        labels.append(1 if i < j else 0)
            sol_line += '\n'
            sol_line_tasks += '\n'

        print(sol_line_tasks)
        print('Task Time Intervals\n')
        print(sol_line)
        return disjunctive_raw_features, disjunctive_combine_features, labels, optimal_obj, seed




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='JSS solver.')
    parser.add_argument('-nmachine', default=10, help='number of machines')
    parser.add_argument('-njob', default=10, type=int, help='number of jobs')
    parser.add_argument('-seed', default=10, type=int, help='random seed')
    args = parser.parse_args()
    nmachine = args.nmachine
    njob = args.njob

    df_raw = pd.DataFrame()
    df_combine = pd.DataFrame()
    df_other = pd.DataFrame()
    # for seed in range(100):
    for seed in range(1000,2000):
        disjunctive_raw_features, disjunctive_combine_features, labels, optobj, _seed = solve_jss_ortools(nmachines=nmachine, njobs=njob, seed=args.seed + 99*seed)
        size = len(labels)
        data_raw = np.hstack((np.array(disjunctive_raw_features).reshape(size, -1), np.array(labels).reshape(size, -1)))
        data_combine = np.hstack((np.array(disjunctive_combine_features).reshape(size, -1), np.array(labels).reshape(size, -1)))
        data_other = np.hstack(((np.ones(size)*optobj).reshape(size, -1), (np.ones(size)*_seed).reshape(size, -1),
                                (np.ones(size)*nmachine).reshape(size, -1), (np.ones(size)*njob).reshape(size, -1)))
        df_raw = df_raw.append(pd.DataFrame(data_raw))
        df_combine = df_combine.append(pd.DataFrame(data_combine))
        df_other = df_other.append(pd.DataFrame(data_other))
        print("Complete solving instance #{}".format(args.seed + 99*seed))

    df_raw.columns = ["opid1", "opid2", "duration1", "duration2", "upstreamdur1", "upstreamdur2", "downstreamdur1", "downstreamdur2", "totaljobdur1", "totaljobdur2", "label"]
    df_combine.columns = ["opid_diff", "duration_diff", "upstreamdur_diff", "downstreamdur_diff", "totaljobdur_diff", "label"]
    df_other.columns = ["optimal_objective", "seed", "nmachine", "njob"]
    df_raw.to_csv("../results/datajss_raw_nm_{}_nj_{}.csv".format(nmachine, njob), index=False)
    df_combine.to_csv("../results/datajss_combine_nm_{}_nj_{}.csv".format(nmachine, njob), index=False)
    df_other.to_csv("../results/datajss_other_nm_{}_nj_{}.csv".format(nmachine, njob), index=False)

