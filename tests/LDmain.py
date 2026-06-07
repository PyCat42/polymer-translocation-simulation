"""
Demonstrating the use of simulation and parallelization classes.
"""

import time

from src.Parallelization import Parallelization

if __name__ == "__main__":
    # running simulation and creating plots that capture each stage
    # LangevinDynamicsSimulation(N=10).simulate_escape(do_plot=True)
    # LangevinDynamicsSimulation(N=10).simulate_translocation(do_plot=True)

    # creating animations
    # LangevinDynamicsSimulation(N=10).animate_escape("LD_escape.gif")
    # LangevinDynamicsSimulation(N=10).animate_translocation("LD_translocation.gif")

    # running parallel simulation
    Parallelization.LJ_numba_warmup()

    sim = Parallelization(num_process=10)

    # start timer
    start = time.perf_counter()

    sim.run_parallel_mp("LD", "escape", [10], [1000],
                        filename="LD_E0_N10_1000samples.npy")

    end = time.perf_counter()

    # benchmarking
    elapsed = end - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(formatted)
