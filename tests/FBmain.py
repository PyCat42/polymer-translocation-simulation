from src.FluctuatingBondSimulation import FluctuatingBondSimulation
from src.Parallelization import Parallelization
import time

if __name__ == "__main__":
    # running simulation and creating plots that capture each stage
    # FluctuatingBondSimulation(N=10).simulate_translocation(do_plot=True)
    # FluctuatingBondSimulation(N=10).simulate_escape(do_plot=True)

    # creating animations
    # FluctuatingBondSimulation(N=10).animate_translocation("FB_translocation.gif")
    # FluctuatingBondSimulation(N=10).animate_escape("FB_escape.gif")

    sim = Parallelization(num_process=10)

    # start timer
    start = time.perf_counter()

    sim.run_parallel_mp("FB", "escape", [10], [1000],
                        filename="FB_E0_N10_1000samples.npy")

    end = time.perf_counter()

    # benchmarking
    elapsed = end - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(formatted)
