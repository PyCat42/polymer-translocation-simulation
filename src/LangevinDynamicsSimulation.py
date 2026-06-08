import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit


# numba functions can provide some additional speedup for simulation
@njit
def LJ_force_numba(r_mon, sigma, epsilon, r_c, out_forces):
    """
    Calculate LJ force between EACH PAIR of beads.
    :return:
    """
    N = r_mon.shape[0]

    for i in range(N):
        x_i = r_mon[i, 0]
        y_i = r_mon[i, 1]

        for j in range(i + 1, N):
            dx = x_i - r_mon[j, 0]
            dy = y_i - r_mon[j, 1]

            r = np.sqrt(dx ** 2 + dy ** 2)

            if 0 < r < r_c:
                ratio = sigma / r
                ratio6 = ratio ** 6

                magnitude = 24 * (epsilon / r) * (2 * ratio6 ** 2 - ratio6)

                f_x = magnitude * dx / r
                f_y = magnitude * dy / r

                out_forces[i, 0] += f_x
                out_forces[i, 1] += f_y

                out_forces[j, 0] -= f_x
                out_forces[j, 1] -= f_y


@njit
def wall_LJ_force_numba(r_mon, wall_beads, sigma, epsilon, r_c, out_forces):
    """
    Calculate LJ force asserted on each monomer by all wall beads.
    This function is optimized to take into account only the beads that are closest to the polymer.
    :return:
    """
    N = r_mon.shape[0]
    W = wall_beads.shape[0]

    for i in range(N):
        x_i = r_mon[i, 0]
        y_i = r_mon[i, 1]

        for j in range(W):
            dx = x_i - wall_beads[j, 0]
            dy = y_i - wall_beads[j, 1]

            r = np.sqrt(dx ** 2 + dy ** 2)

            if 0 < r < r_c:
                ratio = sigma / r
                ratio6 = ratio ** 6

                magnitude = 24 * (epsilon / r) * (2 * ratio6 ** 2 - ratio6)

                out_forces[i, 0] += magnitude * dx / r
                out_forces[i, 1] += magnitude * dy / r


@njit
def FENE_force_numba(r_mon, R_0, k, out_forces):
    """
    Calculate FENE force between each pair of CONSECUTIVE beads.
    :return:
    """
    N = r_mon.shape[0]

    for i in range(N - 1):
        dx = r_mon[i + 1, 0] - r_mon[i, 0]
        dy = r_mon[i + 1, 1] - r_mon[i, 1]

        r = np.sqrt(dx ** 2 + dy ** 2)

        if r < 1e-6: r = 1e-6
        if r > R_0 - 1e-6: r = R_0 - 1e-6

        denom = 1 - (r / R_0) ** 2

        magnitude = k * r / denom

        f_x = magnitude * dx / r
        f_y = magnitude * dy / r

        out_forces[i, 0] += f_x
        out_forces[i, 1] += f_y

        out_forces[i + 1, 0] -= f_x
        out_forces[i + 1, 1] -= f_y


@njit
def Langevin_integration_step_numba(r_mon, v_mon, F, factor, factor_plus, factor_minus, betta, B1, B2, dt):
    """
    Integrate Langevin equation, i.e. update positions and velocities of each monomer bead
    using the method described in Ermak & Buckholz.
    :return:
    """
    N = r_mon.shape[0]

    for i in range(N):
        # velocity update - formula (7a) in Ermak & Buckholz
        v_x_updated = v_mon[i, 0] * factor + F[i, 0] * factor_minus + B1[i, 0]
        v_y_updated = v_mon[i, 1] * factor + F[i, 1] * factor_minus + B1[i, 1]

        # position update - formula (7b) in Ermak & Buckholz
        # both new and old velocity are needed
        r_mon[i, 0] += ((1 / betta) * (v_x_updated + v_mon[i, 0] - 2 * F[i, 0]) * (factor_minus / factor_plus)
                       + F[i, 0] * dt + B2[i, 0])
        r_mon[i, 1] += ((1 / betta) * (v_y_updated + v_mon[i, 1] - 2 * F[i, 1]) * (factor_minus / factor_plus)
                        + F[i, 1] * dt + B2[i, 1])

        # replace odl velocity with updated
        v_mon[i, 0] = v_x_updated
        v_mon[i, 1] = v_y_updated


# ------------------------------------------------------------------------------------------------

class LangevinDynamicsSimulation:
    """
    Class for simulation of polymer translocation using the Langevin equation:
    m(d^2r_i/dt^2) = F_Ci + F_Fi + F_Ri

    Here:

    - conservative force F_Ci = F_LJ + F_FENE + F_driving
    - frictional force F_Fi = - ksi * v_i
    - random force F_Ri is described using fluctuation dissipation theorem:
      - avg(F_Ri(t)) = 0
      - avg(F_Ri(t) * F_Ri(t')) = 6 * kB * T * ksi * delta_ij * delta(t - t')

    **Integration** of this equation is performed using the method from:

    Donald L. Ermak, Helen Buckholz,
    Numerical integration of the Langevin equation: Monte Carlo simulation,
    Journal of Computational Physics, Volume 35, Issue 2, 1980

    **Polymer:**

    - modeled as bead-spring chains with beads of diameter sigma
    - excluded volume and van der Waals interactions between beads
    are modeled by a repulsive Lennard-Jones potential with depth epsilon
    - the connectivity between beads is modeled as Finite Extension Nonlinear Elastic potential spring
    with spring konstant k and maximum allowed separation between beads R_0

    **Wall:**

    - l columns of stationary particles within distance sigma from one another
    - interacting with the beads by the repulsive Lennard-Jones potential

    **Pore:**

    - modeled by removing w beads from each column of the wall

    ***This model is based on:***

    Ilkka Huopaniemi, Kaifu Luo, Tapio Ala-Nissila, See-Chen Ying;
    Langevin dynamics simulations of polymer translocation through nanopores.
    J. Chem. Phys. 28 September 2006; 125 (12): 124901. https://doi.org/10.1063/1.2357118

    """
    def __init__(self, N, m=1.0, l=1.0, w=2.0, sigma=1.0, R_0=2.0,
                 epsilon=1.0, k=7.0, E=0.0, dt=1e-3, ksi=0.7, kBT=1.0):
        self.N = N  # number of monomers in the polymer
        self.m = m  # monomer's mass
        self.r_mon = np.zeros((N, 2))  # list of monomer coordinates
        self.v_mon = np.zeros((N, 2))  # list of monomer velocities

        self.sigma = sigma  # diameter of the bead; fixes system length unit
        self.r_c = 2 ** (1 / 6) * sigma  # critical radius for LJ interaction
        self.R_0 = R_0  # maximum allowed separation between the beads

        self.l = l  # length of the pore
        # define pore entrance and exit
        if l % 2 == 0:
            self.pore_entrance = - np.floor(self.l / 2) * self.sigma
            self.pore_exit = np.floor(self.l / 2) * self.sigma
        else:
            self.pore_entrance = - ((self.l - 1) / 2) * self.sigma
            self.pore_exit = ((self.l - 1) / 2) * self.sigma
        wall_x = np.arange(self.pore_entrance, 0.1 + self.pore_exit, self.sigma)

        self.w = w  # width of the pore
        self.wall_extent = 50.0  # height of the wall (could be larger for larger N)
        # aranging wall beads
        if l % 2 == 0:
            wall_y = np.arange(- self.wall_extent + 0.5 * self.sigma, self.wall_extent, self.sigma)
            pore_mask = np.abs(wall_y) <= (self.w / 2.0)
            wall_y = wall_y[~pore_mask]
        else:
            wall_y = np.arange(- self.wall_extent, self.wall_extent, self.sigma)
            pore_mask = np.abs(wall_y) <= (self.w / 2.0)
            wall_y = wall_y[~pore_mask]
        wall_beads = []
        for x in wall_x:
            for y in wall_y:
                wall_beads.append([x, y])
        self.wall_beads = np.array(wall_beads)

        self.epsilon = epsilon  # parameter adjusting LJ potential depth; fixes system energy unit
        self.k = k  # spring constant

        self.E = E  # applied electrical field
        self.ksi = ksi  # friction coefficient
        self.kBT = kBT

        self.dt = dt  # time step for the simulation

        # some helper factors
        self.betta = self.ksi / self.m
        self.factor = np.exp(-self.betta * self.dt)
        self.factor_plus = 1 + self.factor
        self.factor_minus = 1 - self.factor

        # precalculate velocity and position noise distributions
        # drop the 3 factor since we only need 1D distribution
        self.sigma_B1 = np.sqrt((self.kBT / self.m) * (1 - self.factor**2))
        self.sigma_B2 = np.sqrt((2 * self.kBT / (self.m * self.betta**2))
                                * (self.betta * self.dt - 2 * self.factor_minus / self.factor_plus))
        self.rng = np.random.default_rng()

        self.is_equilibration = False  # flag for equilibration process

        # some buffer sequences to be filled later
        # this helps to avoid creating multiple sequence instances
        # thus saving memory and speeding up simulation
        self.force_buffer = np.zeros((self.N, 2))
        self.B1_buffer = np.zeros((self.N, 2))
        self.B2_buffer = np.zeros((self.N, 2))

    # ---------------------------------------------------------------------------------------------------------------
    # Calculate forces and perform Langevin equation integration
    # - bellow are both regular numpy functions and functions using numba
    #   (although only later are used former are left here as legacy functions and are also fully functional)
    # ---------------------------------------------------------------------------------------------------------------

    def LJ_force(self):
        """
        Calculate LJ force between EACH PAIR of beads.
        :return: array containing calculated LJ force for each monomer in polymer
        """
        r_ij = self.r_mon[:, None, :] - self.r_mon[None, :, :]  # shape (N, N, 2)
        r = np.linalg.norm(r_ij, axis=2)  # shape (N, N)

        r_c = 2**(1 / 6) * self.sigma
        mask = (r > 0) & (r < r_c)

        # exclude self-coupling
        mask[np.eye(self.N, dtype=bool)] = False  # np.eye finds diagonal elements

        # calculate FENE force for the remaining elements
        ratio = np.zeros_like(r)
        ratio[mask] = self.sigma / r[mask]

        magnitude = np.zeros_like(r)
        magnitude[mask] = 24 * (self.epsilon / r[mask]) * (2 * ratio[mask]**12 - ratio[mask]**6)

        f_ij = np.zeros_like(r_ij)
        f_ij[mask] = magnitude[mask][:, None] * (r_ij[mask] / r[mask][:, None])
        # because magnitude is (M,), rij is (M, 2) and r is (M,)
        # so we need to turn magnitude and r into (M, 1)

        f_particle = np.sum(f_ij, axis=1)

        return f_particle

    def fast_LJ_force(self):
        """
        Calculate LJ force between EACH PAIR of beads.
        :return:
        """
        LJ_force_numba(self.r_mon, self.sigma, self.epsilon, self.r_c, self.force_buffer)

    # -----------------------------------------------------------------------------------------------

    def wall_LJ_force(self):
        """
        Calculate LJ force asserted on each monomer by all wall beads.
        This function is optimized to take into account only the beads that are closest to the polymer.
        :return: array containing calculated LJ force coming from interaction with the wall
        for each monomer in polymer
        """
        r_c = 2 ** (1 / 6) * self.sigma  # cutoff radius

        # find monomers with smallest and largest coordinates
        # these will define the effective radius
        polymer_min = np.min(self.r_mon, axis=0) - r_c
        polymer_max = np.max(self.r_mon, axis=0) + r_c

        # wall beads are acting significantly on the monomer if they are in this effective radius
        in_range = ((self.wall_beads[:, 0] >= polymer_min[0]) &
                    (self.wall_beads[:, 0] <= polymer_max[0]) &
                    (self.wall_beads[:, 1] >= polymer_min[1]) &
                    (self.wall_beads[:, 1] <= polymer_max[1]))
        wall_beads_in_range = self.wall_beads[in_range]

        # if there are no beads in the effective radius, this force is 0
        if len(wall_beads_in_range) == 0:
            return np.zeros_like(self.r_mon)

        # create matrix of bonds between all pairs of monomer and wall beads
        r_ij = self.r_mon[:, None, :] - wall_beads_in_range[None, :, :]  # size (N, W_eff, 2)
        r = np.linalg.norm(r_ij, axis=2)

        mask = (r > 0) & (r < r_c)

        ratio = np.zeros_like(r)
        ratio[mask] = self.sigma / r[mask]

        magnitude = np.zeros_like(r)
        magnitude[mask] = 24 * (self.epsilon / r[mask]) * (2 * ratio[mask] ** 12 - ratio[mask] ** 6)

        f_ij = np.zeros_like(r_ij)
        f_ij[mask] = magnitude[mask][:, None] * (r_ij[mask] / r[mask][:, None])

        # sum calculated forces for each monomer particle
        f_particle = np.sum(f_ij, axis=1)

        return f_particle

    def fast_wall_LJ_force(self):
        """
        Calculate LJ force asserted on each monomer by all wall beads.
        This function is optimized to take into account only the beads that are closest to the polymer.
        :return:
        """
        # find monomers with smallest and largest coordinates
        # these will define the effective radius
        polymer_min = np.min(self.r_mon, axis=0) - self.r_c
        polymer_max = np.max(self.r_mon, axis=0) + self.r_c

        # wall beads are acting significantly on the monomer if they are in this effective radius
        in_range = ((self.wall_beads[:, 0] >= polymer_min[0]) &
                    (self.wall_beads[:, 0] <= polymer_max[0]) &
                    (self.wall_beads[:, 1] >= polymer_min[1]) &
                    (self.wall_beads[:, 1] <= polymer_max[1]))
        wall_beads_in_range = self.wall_beads[in_range]

        wall_LJ_force_numba(self.r_mon, wall_beads_in_range, self.sigma, self.epsilon, self.r_c, self.force_buffer)

    # -----------------------------------------------------------------------------------------------

    def FENE_force(self):
        """
        Calculate FENE force between each pair of CONSECUTIVE beads.
        :return: array containing calculated FENE force for each monomer in polymer
        """
        # find bond vectors and bond lengths between consecutive vectors
        r_ij = self.r_mon[1:] - self.r_mon[:-1]
        r = np.linalg.norm(r_ij, axis=1)

        # make sure values of r are above 0 and bellow R_0 at all times
        # ensuring bonds don't snap and maximum bond length is enforced
        safe_r = np.clip(r, 0.001, self.R_0 - 1e-6)
        denom = 1 - (safe_r / self.R_0)** 2

        # calculate magnitude and force
        magnitude = self.k * safe_r / denom
        f_ij = magnitude[:, None] * (r_ij / safe_r[:, None])

        # sum contributions for each particle
        f_particle = np.zeros_like(self.r_mon)
        f_particle[:-1] += f_ij
        f_particle[1:] -= f_ij

        return f_particle

    def fast_FENE_force(self):
        """
        Calculate FENE force between each pair of CONSECUTIVE beads.
        :return:
        """
        FENE_force_numba(self.r_mon, self.R_0, self.k, self.force_buffer)

    # -----------------------------------------------------------------------------------------------

    def driving_force(self):
        """
        Calculate the driving force asserted on the monomer bead by the pore
        across which an electric field of magnitude E is applied.
        :return:
        """
        # this ensures there is no driving force during equilibration
        if not self.is_equilibration:
            x = self.r_mon[:, 0]
            y = self.r_mon[:, 1]

            mask = (np.abs(x) <= self.l / 2) & (np.abs(y) <= self.w / 2)

            self.force_buffer[mask] += np.array([self.E, 0])

    # -----------------------------------------------------------------------------------------------

    def Langevin_integration_step(self):
        """
        Integrate Langevin equation, i.e. update positions and velocities of each monomer bead
        using the method described in Ermak & Buckholz.
        :return:
        """
        self.force_buffer.fill(0.0)

        self.fast_LJ_force()
        self.fast_wall_LJ_force()
        self.fast_FENE_force()
        self.driving_force()

        self.force_buffer /= self.ksi  # this is used in formulas for updating r and v

        # sample random noise distributions for position and velocity
        # these are not correlated if formulas (7a) and (7b) are used for position and velocity updates.
        B1 = np.random.normal(loc=0.0, scale=self.sigma_B1, size=(self.N, 2))
        B2 = np.random.normal(loc=0.0, scale=self.sigma_B2, size=(self.N, 2))

        # velocity update - formula (7a) in Ermak & Buckholz
        v_updated = self.v_mon * self.factor + self.force_buffer * self.factor_minus + B1

        # position update - formula (7b) in Ermak & Buckholz
        # both new and old velocity are needed
        self.r_mon += ((1 / self.betta) * (v_updated + self.v_mon - 2 * self.force_buffer)
                       * (self.factor_minus / self.factor_plus)
                       + self.force_buffer * self.dt + B2)

        # replace odl velocity with updated
        self.v_mon = v_updated

    def fast_Langevin_integration_step(self):
        """
        Integrate Langevin equation, i.e. update positions and velocities of each monomer bead
        using the method described in Ermak & Buckholz.
        :return:
        """
        self.force_buffer.fill(0.0)

        self.fast_LJ_force()
        self.fast_wall_LJ_force()
        self.fast_FENE_force()
        self.driving_force()

        self.force_buffer /= self.ksi  # this is used in formulas for updating r and v

        self.B1_buffer[:] = self.rng.normal(loc=0.0, scale=self.sigma_B1, size=(self.N, 2))
        self.B2_buffer[:] = self.rng.normal(loc=0.0, scale=self.sigma_B2, size=(self.N, 2))

        Langevin_integration_step_numba(self.r_mon, self.v_mon, self.force_buffer,
                                        self.factor, self.factor_plus, self.factor_minus, self.betta,
                                        self.B1_buffer, self.B2_buffer, self.dt)

    # ---------------------------------------------------------------------------------------------------------------
    # Simulate translocation
    # ---------------------------------------------------------------------------------------------------------------

    def translocation_equilibration(self):
        """
        Let polymer relax for a time larger than Rose translocation_equilibration time (~ N^2)
        by applying Langevin integration to all monomers except to the first one
        (that is fixed at the pore entrance after each integration step)
        to obtain the initial configuration.
        :return:
        """
        self.is_equilibration = True

        steps = int(self.N**2.5 / self.dt)
        for _ in range(steps):
            self.fast_Langevin_integration_step()

            # keep first monomer fixed at the pore entrance
            self.r_mon[0, 0] = self.pore_entrance
            self.r_mon[0, 1] = 0.0
            self.v_mon[0, :] = 0.0

        self.is_equilibration = False

    def translocation_stopping_condition(self):
        """
        In the case of polymer translocation, the first monomer of the polymer is initially placed
        at the beginning of the pore. Polymer is then released and can translocate to the other side of the wall.
        When all monomers of the polymer reach the other side of the wall
        the monomer is said to have successfully translocated through the pore.
        In case all the monomers are to the left, the translocation has failed.
        :return:
        """
        if np.all(self.r_mon[:, 0] > self.l / 2):
            return "success"
        elif np.all(self.r_mon[:, 0] < - self.l / 2):
            return "failure"
        else:
            return "running"

    def simulate_translocation(self, do_plot=False):
        """
        Main loop that performs translocation simulation.
        :param do_plot: if True plot snapshots of the proces
        :return: translocation time i.e. time in which polymer managed to translocate out of the pore
                (move all monomers to the opposite side of the wall)
        """
        # create simple linear monomer with first monomer at the entrance of the pore
        self.r_mon = np.zeros((self.N, 2))
        bond_length = 1.1 * self.sigma
        self.r_mon[:, 0] = self.pore_entrance - np.arange(bond_length * self.N, step=bond_length)

        if do_plot:
            self.plot_simulation("start.png")

        # let it equilibrate
        self.translocation_equilibration()

        if do_plot:
            self.plot_simulation("../equilibrated.png")

        # release the monomers and let polymer translocate through the pore
        # track the translocation time
        transloc_t = 0
        # track translocation status
        status = "running"
        # track times in which monomers manage to exit the pore
        exit_times = np.full(self.N, -1.0)
        while status == "running":
            self.fast_Langevin_integration_step()
            transloc_t += 1

            exited = self.r_mon[:, 0] > self.pore_exit
            not_recorded = exit_times < 0
            to_update = exited & not_recorded
            exit_times[to_update] = transloc_t

            status = self.translocation_stopping_condition()

            if transloc_t % 500 == 0 and do_plot:
                self.plot_simulation(f"translocation{transloc_t}.png")

        if do_plot:
            self.plot_simulation("../end.png")

        if status == "failure":
            return -1.0
        else:
            # waiting_times = np.concatenate(([exit_times[0]], np.diff(exit_times)))
            waiting_times = np.empty_like(exit_times)
            waiting_times[0] = 0
            waiting_times[1:] = exit_times[1:] - exit_times[:-1]
            return transloc_t * self.dt, waiting_times * self.dt

    # ---------------------------------------------------------------------------------------------------------------
    # Simulate escape
    # ---------------------------------------------------------------------------------------------------------------

    def escape_equilibration(self):
        """
        Let polymer relax for a time larger than Rose translocation_equilibration time (~ N^2)
        by applying Langevin integration to all monomers except to the middle monomers
        (that are fixed inside the pore after each integration step)
        to obtain the initial configuration.
        :return:
        """
        self.is_equilibration = True

        # fix monomers in the middle of the pore so they can't escape during equilibration
        pinned_monomers = [i for i in range(self.N)
                           if self.pore_entrance <= self.r_mon[i, 0] <= self.pore_exit]
        if not pinned_monomers:
            if self.N % 2 == 0:
                pinned_monomers = [self.N // 2 - 1, self.N // 2]
            else:
                pinned_monomers = [self.N // 2]

        pinned_locs = self.r_mon[pinned_monomers, :].copy()

        steps = int(self.N ** 2.5 / self.dt)
        for _ in range(steps):
            self.fast_Langevin_integration_step()

            # keep first monomer fixed at the pore entrance
            for i, pinned in enumerate(pinned_monomers):
                self.v_mon[pinned] = np.array([0, 0])
                self.r_mon[pinned] = pinned_locs[i]

        self.is_equilibration = False

    def escape_stopping_condition(self):
        """
        In the case of polymer escape, the middle pof the polymer is initially placed
        inside the pore. Polymer is then released and can escape to either of the sides.
        When all monomers of the polymer reach one side of the wall
        the monomer is said to have escaped the pore.
        :return:
        """
        all_right = np.all(self.r_mon[:, 0] > self.pore_exit)
        all_left = np.all(self.r_mon[:, 0] < self.pore_entrance)
        return all_right or all_left

    def simulate_escape(self, do_plot=False):
        """
        Main loop that performs escape simulation.
        :param do_plot: if True plot snapshots of the proces
        :return: escape time i.e. time in which polymer managed to escape
        out of the pore to the either side of the wall
        """
        # create simple linear polymer with middle monomers inside the pore
        # pore center is at (0, 0)
        self.r_mon = np.zeros((self.N, 2))
        spacings = np.arange(self.N) - (self.N - 1) / 2
        bond_length = 1.1 * self.sigma
        self.r_mon[:, 0] = bond_length * spacings[::-1]

        if do_plot:
            self.plot_simulation("escape_start.png")

        # let it equilibrate
        self.escape_equilibration()

        if do_plot:
            self.plot_simulation("escape_equilibrated.png")

        # release the monomers and let polymer escape through the pore
        # track the escape time
        esc_t = 0
        while not self.escape_stopping_condition():
            self.fast_Langevin_integration_step()
            esc_t += 1

            if esc_t % 1000 == 0 and do_plot:
                self.plot_simulation(f"escape_translocation{esc_t}.png")

        if do_plot:
            self.plot_simulation("escape_end.png")

        return esc_t * self.dt

    # ---------------------------------------------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------------------------------------------

    def plot_simulation(self, filename):
        """
        Plots simulation snapshots.
        :param filename: name of the file to which snapshot is saved
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # draw wall:
        x_wall = self.wall_beads[:, 0]
        y_wall = self.wall_beads[:, 1]
        ax.scatter(x_wall, y_wall, color='lightblue', edgecolor='blue', s=30)

        # draw polymer:
        x_mon = self.r_mon[:, 0]
        y_mon = self.r_mon[:, 1]
        # - bonds
        ax.plot(x_mon, y_mon, color='black', alpha=0.6, linewidth=1)
        # - monomers
        ax.scatter(x_mon, y_mon, c=np.arange(self.N), cmap='viridis_r', edgecolor='black', s=30)

        # arrange final plot:
        ax.set_aspect('equal')
        ax.set_xlim(-self.N, self.N)
        ax.set_ylim(-self.N, self.N)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.savefig(filename, dpi=300)
        plt.close(fig)

    # ---------------------------------------------------------------------------------------------------------------
    # Animations
    # ---------------------------------------------------------------------------------------------------------------

    def animate_translocation(self, filename="escape.gif"):
        """
        Creates a gif of the polymer translocation simulation.
        :param filename: name of the file to which gif is saved
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Langevin Dynamics Simulation")

        # wall setup:
        # draw wall:
        x_wall = self.wall_beads[:, 0]
        y_wall = self.wall_beads[:, 1]
        ax.scatter(x_wall, y_wall, color='lightblue', edgecolor='blue', s=30)

        # polymer setup:
        # - bonds
        line, = ax.plot([], [], color='black', alpha=0.6, linewidth=1)
        # - monomers
        dots = ax.scatter([], [], c=[], cmap='viridis_r', vmin=0, vmax=self.N, edgecolor='black', s=30)

        # text overlay setup
        state_text = ax.text(0.05, 0.95, '',
                             va='top', ha='left',
                             transform=ax.transAxes, fontsize=12, fontweight='bold')

        # create simple linear monomer with first monomer at the entrance of the pore
        self.r_mon = np.zeros((self.N, 2))
        bond_length = 1.1 * self.sigma
        self.r_mon[:, 0] = self.pore_entrance - np.arange(bond_length * self.N, step=bond_length)

        # initialize graphics
        def init():
            ax.set_aspect('equal')
            ax.set_xlim(-self.N, self.N)
            ax.set_ylim(-self.N, self.N)
            return line, dots

        # track frames for different phases of the process
        def frame_generator():
            # Phase 1: Equilibration (50 frames, Rouse time)
            for f in range(50):
                yield ('Equilibration', f)

            # Phase 2: Translocation
            count = 0
            while self.translocation_stopping_condition() == "running":
                yield ('Translocation', count)
                count += 1
                # safety break
                if count > 5000: break

            # Phase 3: Escape complete
            for f in range(25):
                yield ('Done', f)

        # update image for every frame
        def update(gen_data):
            phase, frame_num = gen_data
            status = self.translocation_stopping_condition()

            # Phase 1: Equilibration (50 frames, Rouse time)
            if phase == 'Equilibration':
                state_text.set_text(
                    f'Equilibration\n'
                    f'N = {self.N}\n'
                    f'E = {self.E}'
                )
                self.is_equilibration = True
                # run this step for 50 frames
                steps_per_frame = int((self.N**2.5) / (50 * self.dt))
                for _ in range(steps_per_frame):
                    self.fast_Langevin_integration_step()

                    self.r_mon[0] = np.array([self.pore_entrance, 0.0])
                    self.v_mon[0] = np.array([0.0, 0.0])

            # Phase 2: Translocation
            elif phase == 'Translocation':
                state_text.set_text(
                    f'Translocation\n'
                    f'N = {self.N}\n'
                    f'E = {self.E}'
                )
                self.is_equilibration = False
                for _ in range(1000):
                    if self.translocation_stopping_condition() == 'running':
                        self.fast_Langevin_integration_step()

            # Phase 3: Escape complete
            else:
                final_status = self.translocation_stopping_condition()
                if final_status == "failure":
                    state_text.set_text('Translocation Failed')
                else:
                    state_text.set_text('Translocation Completed Successfully')

            # plot
            line.set_data(self.r_mon[:, 0], self.r_mon[:, 1])
            dots.set_offsets(self.r_mon)
            dots.set_array(np.arange(self.N))

            return line, dots, state_text

        # forward to animation function and save the gif
        anim = FuncAnimation(fig, update, frames=frame_generator, init_func=init, blit=True, save_count=1000)
        anim.save(filename, writer=PillowWriter(fps=20))
        plt.close()
        print("Animation saved")

    def animate_escape(self, filename="escape.gif"):
        """
        Creates a gif of the polymer escape simulation.
        :param filename: name of the file to which gif is saved
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Langevin Dynamics Simulation")

        # wall setup:
        # draw wall:
        x_wall = self.wall_beads[:, 0]
        y_wall = self.wall_beads[:, 1]
        ax.scatter(x_wall, y_wall, color='lightblue', edgecolor='blue', s=30)

        # polymer setup:
        # - bonds
        line, = ax.plot([], [], color='black', alpha=0.6, linewidth=1)
        # - monomers
        dots = ax.scatter([], [], c=[], cmap='viridis_r', vmin=0, vmax=self.N, edgecolor='black', s=30)

        # text overlay setup
        state_text = ax.text(0.05, 0.95, '',
                             va='top', ha='left',
                             transform=ax.transAxes, fontsize=12, fontweight='bold')

        # create simple linear monomer with first monomer at the entrance of the pore
        self.r_mon = np.zeros((self.N, 2))
        spacings = np.arange(self.N) - (self.N - 1) / 2
        bond_length = 1.1 * self.sigma
        self.r_mon[:, 0] = bond_length * spacings[::-1]

        pinned_monomers = [i for i in range(self.N)
                           if self.pore_entrance <= self.r_mon[i, 0] <= self.pore_exit]

        if not pinned_monomers:
            if self.N % 2 == 0:
                pinned_monomers = [self.N // 2 - 1, self.N // 2]
            else:
                pinned_monomers = [self.N // 2]

        pinned_locs = self.r_mon[pinned_monomers].copy()

        # initialize graphics
        def init():
            ax.set_aspect('equal')
            ax.set_xlim(-self.N, self.N)
            ax.set_ylim(-self.N, self.N)
            return line, dots

        # track frames for different phases of the process
        def frame_generator():
            # Phase 1: Equilibration (50 frames, Rouse time)
            for f in range(50):
                yield ('Equilibration', f)

            # Phase 2: Escape
            count = 0
            while not self.escape_stopping_condition():
                yield ('Escape', count)
                count += 1
                # safety break
                if count > 10e9: break

            # Phase 3: Escape complete
            for f in range(10):
                yield ('Done', f)

        # update image for every frame
        def update(gen_data):
            phase, frame_num = gen_data

            # Phase 1: Equilibration (50 frames, Rouse time)
            if phase == 'Equilibration':
                state_text.set_text(
                    f'Equilibration\n'
                    f'N = {self.N}\n'
                    f'E = {self.E}'
                )
                self.is_equilibration = True

                # run this step for 50 frames
                steps_per_frame = int((self.N**2.5) / (50 * self.dt))
                for _ in range(steps_per_frame):
                    self.fast_Langevin_integration_step()

                    for i, pinned in enumerate(pinned_monomers):
                        self.v_mon[pinned] = np.array([0, 0])
                        self.r_mon[pinned] = pinned_locs[i]

            # Phase 2: Escape
            elif phase == 'Escape':
                state_text.set_text(
                    f'Escape\n'
                    f'N = {self.N}\n'
                    f'E = {self.E}'
                )
                self.is_equilibration = False
                for _ in range(1000):
                    if not self.escape_stopping_condition():
                        self.fast_Langevin_integration_step()

            # Phase 3: Escape complete
            else:
                state_text.set_text('Escape Completed')

            # plot
            line.set_data(self.r_mon[:, 0], self.r_mon[:, 1])
            dots.set_offsets(self.r_mon)
            dots.set_array(np.arange(self.N))

            return line, dots, state_text

        # forward to animation function and save the gif
        anim = FuncAnimation(fig, update, frames=frame_generator, init_func=init, blit=True, save_count=1000)
        anim.save(filename, writer=PillowWriter(fps=20))
        plt.close()
        print("Animation saved")
