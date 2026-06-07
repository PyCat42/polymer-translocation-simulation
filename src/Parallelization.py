import random
import time

import numpy as np
import multiprocessing as mp

from src.FluctuatingBondSimulation import FluctuatingBondSimulation
from src.LangevinDynamicsSimulation import (LangevinDynamicsSimulation, LJ_force_numba, wall_LJ_force_numba,
                                            FENE_force_numba, Langevin_integration_step_numba)


class Parallelization:
    """
    Enables running simulations concurrently.
    """

    def __init__(self, num_process=4, machine_id=1606.2021):
        self.num_process = num_process  # number of processes to run the simulation on
        self.machine_id = machine_id  # machine id enabling generation of unique seeds across different machines

    @staticmethod
    def LJ_numba_warmup():
        """
        Runs all Langevin dynamics simulations using numba.
        :return:
        """
        r_mon = np.zeros((2, 2))
        v_mon = np.zeros((2, 2))
        F = np.zeros((2, 2))
        wall_beads = np.zeros((2, 2))

        sigma = 1.0
        epsilon = 1.0
        r_c = 1.0

        R_0 = 2.0
        k = 1.0

        factor = 0.9
        factor_plus = 1.9
        factor_minus = 0.1

        betta = 1.0
        dt = 0.001

        B1 = np.zeros((2, 2))
        B2 = np.zeros((2, 2))

        LJ_force_numba(r_mon, sigma, epsilon, r_c, F)
        wall_LJ_force_numba(r_mon, wall_beads, sigma, epsilon, r_c, F)
        FENE_force_numba(r_mon, R_0, k, F)
        Langevin_integration_step_numba(r_mon, v_mon, F, factor, factor_plus, factor_minus, betta, B1, B2, dt)

        print("Numba kernel ready...")

    @staticmethod
    def worker_task(args):
        """
        Code executed by a worker.
        :param args: tuple containing
                    - seed: random seed for worker
                    - sim_type: simulation type (Langevin Dynamics "LD" or Fluctuating Bond "FB")
                    - process: "translocation" or "simulation"
                    - sim_kwargs: additional arguments to pass to the class constructor
        :return: value of N for which the worker ran the simulation
                and the simulation result (escape/translocation time)
        """
        # unpack arguments
        seed, sim_type, process, sim_kwargs = args

        # assign seeds
        random.seed(seed)
        np.random.seed(seed)

        # construct class instance for the right simulation class
        if sim_type == "LD":
            sim = LangevinDynamicsSimulation(**sim_kwargs)
        elif sim_type == "FB":
            sim = FluctuatingBondSimulation(**sim_kwargs)

        # run the simulation of the process
        if process == "translocation":
            result = sim.simulate_translocation(do_plot=False)
        elif process == "escape":
            result = sim.simulate_escape(do_plot=False)

        return sim_kwargs['N'], result

    def run_parallel_mp(self, sim_type, process, N_vals, sim_per_N, filename=None, **kwargs):
        """
        Python multiprocessing lib implementation of concurrent simulation
        for large number of simulation runs for one value of N.
        :param filename: name of the .npy file to which the simulation results will be saved
        :param sim_type: simulation type (Langevin Dynamics "LD" or Fluctuating Bond "FB")
        :param process: "translocation" or "simulation"
        :param N_vals: values of N to simulate
        :param sim_per_N: number of simulations runs for each value of N
        :param kwargs: additional arguments to pass to the class constructor
        :return: - dictionary all_times
                    key = N value
                    value = list of escape times
                 - dictionary all_waiting_times (if process == translocation)
                    key = N value
                    value = list of waiting times per monomer

        """
        # initialize dictionary to store all the process times...
        all_times = {n: [] for n in N_vals}
        #... and all waiting times
        all_waiting_times = {n: [] for n in N_vals}
        # and a list to store all the tasks
        all_tasks = []

        # accumulate all the tasks (i.e. args that will be distributed to the workers)
        for n, num_vals in zip(N_vals, sim_per_N):
            current_kwargs = {'N': n}  # add N val to kwargs for this specific batch
            current_kwargs.update(kwargs)  # add all the other kwargs passed to the function
            # make worker tasks
            for i in range(num_vals):
                task_seed = hash((i, n, self.machine_id, time.time_ns())) % (2**32)
                all_tasks.append((task_seed, sim_type, process, current_kwargs))

        tasks_num = len(all_tasks)
        with mp.Pool(processes=self.num_process) as pool:
            # by using the imap_unordered, each worker will get the task as soon as it becomes available
            # for as long as there are tasks
            # this is extremely important since simulation times can significantly vary
            for n_val, results in pool.imap_unordered(Parallelization.worker_task, all_tasks):
                if process == "translocation":
                    # since we only have waiting times for translocation
                    if isinstance(results, tuple):
                        # in case translocation is successful
                        total_t, waiting_t = results
                        all_times[n_val].append(total_t)
                        all_waiting_times[n_val].append(waiting_t)
                    else:
                        # failed translocation case: results == -1.0
                        all_times[n_val].append(results)
                else:
                    # escape case
                    all_times[n_val].append(results)

                # track the progress
                finished = sum(len(v) for v in all_times.values())
                if finished % 10 == 0:
                    print(f"{finished}/{tasks_num} tasks finished")

        # all results are saved to the numpy dict
        # that can be later combined to obtain sufficient statistics
        to_save = {
            "times": all_times,
            "waiting_times": all_waiting_times if process == "translocation" else None
        }
        if filename is None:
            filename = f"{sim_type}_{process}_times_dict.npy"
        np.save(filename, to_save)
        print(f"{process} times saved to {filename}")

        return to_save
