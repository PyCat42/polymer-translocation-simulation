import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter


class FluctuatingBondSimulation:
    """
    Class for simulation of polymer translocation through the nanopore using
    fluctuating bond model.

    **Polymer Modelling:**
    - 2D lattice FB model for MC simulations of a self-avoiding polymer
    - the bond lengths are allowed to vary in the range 2 (for self-avoidance)
     to sqrt(13) (prevents bonds from crossing each other) in units of the lattice constant
    - each segment separating two adjacent effective monomers has a physical meaning
     corresponding approximately to the Kuhn length a
    - the stiffness of the chain is controlled through an angle-dependent potential
     U/kBT=−Jsum(cos(phi))

     **Pore and Wall**
     - wall: sites on lattice unavailable for monomer moves
     - pore: opening of length l and width w in units of lattice constant in the wall
     - the external driving force was modeled as a potential difference
      applied linearly across the length of the pore

    **Dynamics:**
    Metropolis moves of a single segment, with a probability
    of acceptance min (exp(-deltaU/kBT,1) where deltaU is the energy difference
    between the new and old states
    - elementary MC move:
        - randomly select a monomer
        - attempt to move it onto an adjacent lattice site in a randomly selected direction
        - if the new position does not violate bond-length restrictions, the move is accepted
        or rejected according to Metropolis criterion
    - MC time step: N elementary moves

    Based on:
    Kaifu Luo, T. Ala-Nissila, See-Chen Ying;
    Polymer translocation through a nanopore: A two-dimensional Monte Carlo study.
    J. Chem. Phys. 21 January 2006; 124 (3): 034714. https://doi.org/10.1063/1.2161189
    """

    # bond
    MIN_BOND_LENGTH2 = 4
    MAX_BOND_LENGTH2 = 13

    # possible monomer moves on the lattice
    UP = (1, 0)
    DOWN = (-1, 0)
    RIGHT = (0, 1)
    LEFT = (0, -1)

    def __init__(self, N, L=3.0, w=2.0, J=0, E=5.0, kBT=1.0):
        self.N = N  # number of monomers in the polymer

        self.L = L  # length of the pore
        self.pore_entrance = -self.L / 2
        self.pore_exit = self.L / 2
        self.w = w  # width of the pore
        self.pore_floor = -self.w / 2
        self.pore_ceil = self.w / 2
        self.wall_extent = 1000

        self.J = J  # interaction strength
        self.E = E  # electric field applied across the pore
        self.kBT = kBT  # kB is the Boltzmann constant, and T is the absolute temperature

        self.monomers = np.zeros((N, 2))  # list of monomer coordinates

        self.is_equilibration = False  # flag for equilibration process

    # ------------------------------------------------------------------------------------------------------
    # Check physical limitations...
    # ------------------------------------------------------------------------------------------------------

    def wall_check(self, new_loc):
        """
        Checks if monomer crossed wall boundaries (in which case the move is invalid).
        Wall is located at y in range [-L/2, L/2] for x in range (-1000, 0] U [w, 1000 + w).
        :param new_loc: location to which monomer is moved in one MC move
        :return: True if move is valid, False otherwise
        """
        m_x, m_y = new_loc

        if self.pore_entrance <= m_x <= self.pore_exit:
            return self.pore_floor <= m_y <= self.pore_ceil
        else:
            return True

    def excluded_volume_check(self, new_loc):
        """
        Checks if monomer is in the excluded volume of another monomer
        (in which case the move is invalid).
        :param new_loc: location to which monomer is moved in one MC move
        :return: True if move is valid, False otherwise
        """
        distances = np.sum((self.monomers - new_loc)**2, axis=1)
        if np.sum(distances < self.MIN_BOND_LENGTH2) > 1:
            return False
        else:
            return True

    def bond_length_check(self, mon_num, new_loc):
        """
        Check if bond length is larger than 2 (which accounts for excluded volume effects)
        and smaller than sqrt(13).
        :param mon_num: number of monomer that is chosen to be moved
        :param new_loc: location to which monomer is moved in one MC move
        :return: True if bond length is in given boundaries, False otherwise
        """
        # only check bond length between chosen monomer two of its immediate neighbors...
        #... 1 monomer before
        if mon_num != 0:
            prev_dist = np.sum((self.monomers[mon_num - 1] - new_loc)**2)
            if prev_dist < self.MIN_BOND_LENGTH2 or prev_dist > self.MAX_BOND_LENGTH2:
                return False
        #... and 1 monomer after
        if mon_num != self.N - 1:
            next_dist = np.sum((self.monomers[mon_num + 1] - new_loc)**2)
            if next_dist < self.MIN_BOND_LENGTH2 or next_dist > self.MAX_BOND_LENGTH2:
                return False
        return True

    # ------------------------------------------------------------------------------------------------------
    # Calculate potentials...
    # ------------------------------------------------------------------------------------------------------

    def bond_potential(self, mon_num):
        """
        Stiffness of polymer is modeled using angle dependent potential.
        When calculating this potential we only care about the chosen monomer
        and two of its immediate neighbors.
        :param mon_num: number of monomer that is chosen to be moved
        :return:
        """
        start = max(0, mon_num - 2)
        end = min(self.N - 1, mon_num + 2)
        affected = self.monomers[start:end+1]
        bonds = np.diff(affected, axis=0)
        b1 = bonds[:-1]
        b2 = bonds[1:]
        cos_phi = np.sum(b1 * b2, axis=1) / (np.linalg.norm(b1, axis=1) * np.linalg.norm(b2, axis=1))
        U = -self.J * np.sum(cos_phi)
        return U

    def field_potential(self, new_loc):
        """
        Pore electric field potential of monomer to be moved.
        :param new_loc: location to which monomer is to be moved
        :return:
        """
        if self.is_equilibration:
            # no electric field during equilibration!
            return 0.0
        else:
            # only the potential of the monomer whose location is updated changes
            x = new_loc[0]
            if x < self.pore_entrance:
                return - self.E * self.pore_entrance
            elif x > self.pore_exit:
                return - self.E * self.pore_exit
            else:
                return - self.E * x

    def calculate_potential(self, mon_num, new_loc):
        """
        Calculates total potential (bond + electric field) after a move.
        :param mon_num: number of monomer that was moved
        :param new_loc: location to which chosen monomer was moved
        :return: total potential
        """
        return self.bond_potential(mon_num) + self.field_potential(new_loc)

    # ------------------------------------------------------------------------------------------------------
    # Simulation core...
    # ------------------------------------------------------------------------------------------------------

    def MC_step(self, pinned_monomers=None):
        """
        Perform one MC step
        :param pinned_monomers: monomers that are not allowed to move during the step
        :return:
        """

        if pinned_monomers is None:
            pinned_monomers = []

        # one MC steps consists of N elementary MC steps
        for i in range(self.N):
            # randomly choose one monomer to move
            monomer_to_move = random.randint(0, self.N-1)

            # if it's pinned it is won't be moved
            if monomer_to_move in pinned_monomers:
                continue

            # randomly select a direction in which to move the polymer
            directions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
            move = random.choice(directions)

            # mark old and new location of the selected monomer
            old_loc = self.monomers[monomer_to_move]
            new_loc = self.monomers[monomer_to_move] + move

            # calculate potential of the old configuration
            U_old = self.calculate_potential(monomer_to_move, old_loc)

            # calculate potential of the new configuration and potential difference
            U_new = self.calculate_potential(monomer_to_move, new_loc)
            delta_U = U_new - U_old

            # calculate metropolis weight with which the move is accepted
            metropolis_weight = np.exp(-delta_U / self.kBT)

            # check if physical constraints are satisfied
            if (self.wall_check(new_loc) and self.excluded_volume_check(new_loc)
                    and self.bond_length_check(monomer_to_move, new_loc)):

                if delta_U <= 0:
                    # if potential in the new state is smaller than the potential
                    # in the old state we accept it
                    self.monomers[monomer_to_move] = new_loc
                else:
                    # in the opposite case we can accept/reject the new state
                    # with a probability defined by the Metropolis weight
                    if random.random() > metropolis_weight:
                        self.monomers[monomer_to_move] = old_loc
                    else:
                        self.monomers[monomer_to_move] = new_loc

    def equilibration(self, pinned_monomers):
        """
        Let polymer reach equilibrium state before starting the simulation
        by performing MC steps for a period larger than Rouse equilibration period.
        :param pinned_monomers: monomers that are not allowed to move during the step
        :return:
        """
        self.is_equilibration = True

        # run MC steps
        time = int(20 * self.N ** 2.5)  # Rouse equilibration time ~ N^2
        for t in range(time):
            self.MC_step(pinned_monomers=pinned_monomers)

        self.is_equilibration = False

    # ------------------------------------------------------------------------------------------------------
    # Simulate escape...
    # ------------------------------------------------------------------------------------------------------

    def escape_stopping_condition(self):
        """
        In the case of polymer escape, the middle pof the polymer is initially placed
        inside the pore. Polymer is then released and can escape to either of the sides.
        When all monomers of the polymer reach one side of the wall
        the monomer is said to have escaped the pore.
        :return:
        """
        # are all monomers to the right of the wall?
        all_right = np.all(self.monomers[:, 0] > self.pore_exit)
        # are all monomers to the left of the wall?
        all_left = np.all(self.monomers[:, 0] < self.pore_entrance)
        # if either one is true the monomer has escaped
        return all_right or all_left

    def simulate_escape(self, do_plot=False):
        """
        Main loop that performs escape simulation.
        :param do_plot: if True plot snapshots of the proces
        :return: escape time i.e. MC time in which polymer managed to escape the pore
                (move all monomers to one side of the wall)
        """
        self.monomers = np.zeros((self.N, 2))
        spacings = np.arange(self.N) - (self.N - 1) / 2
        bond_length = 3
        self.monomers[:, 0] = bond_length * spacings[::-1]

        if do_plot:
            self.plot_simulation("escape", "escape_start.png")

        # let it equilibrate - pin the monomers inside the pore
        pinned_monomers = [i for i in range(self.N) if self.pore_entrance <= self.monomers[i, 0] <= self.pore_exit]

        if not pinned_monomers:
            if self.N % 2 == 0:
                pinned_monomers = [self.N // 2 - 1, self.N // 2]
            else:
                pinned_monomers = [self.N // 2]

        self.equilibration(pinned_monomers=pinned_monomers)

        if do_plot:
            self.plot_simulation("escape", "escape_equilibrated.png")

        # release the monomers and let polymer escape the pore
        # track the MC time
        esc_t = 0
        while not self.escape_stopping_condition():
            self.MC_step()
            esc_t += 1

            if esc_t % 500 == 0 and do_plot:
                self.plot_simulation("escape", f"escape{esc_t}.png")

        if do_plot:
            self.plot_simulation("escape", "escape_end.png")

        return esc_t

    # ------------------------------------------------------------------------------------------------------
    # Simulate translocation...
    # ------------------------------------------------------------------------------------------------------

    def translocation_stopping_condition(self):
        """
        In the case of polymer translocation, the first monomer of the polymer is initially placed
        at the beginning of the pore. Polymer is then released and can translocate to the other side of the wall.
        When all monomers of the polymer reach the other side of the wall
        the monomer is said to have successfully translocated through the pore.
        In case all the monomers are to the left, the translocation has failed.
        :return:
        """
        if np.all(self.monomers[:, 0] > self.pore_exit):
            return "success"
        elif np.all(self.monomers[:, 0] < self.pore_entrance):
            return "failure"
        else:
            return "running"

    def simulate_translocation(self, do_plot=False):
        """
        Main loop that performs translocation simulation.
        :param do_plot: if True plot snapshots of the proces
        :return: escape time i.e. MC time in which polymer managed to translocate through the pore
                (move all monomers to right of the wall)
        """
        self.monomers = np.zeros((self.N, 2))
        bond_length = 2
        self.monomers[:, 0] = self.pore_entrance - np.arange(bond_length * self.N, step=bond_length)

        if do_plot:
            self.plot_simulation("translocation", "translocation_start.png")

        # let it equilibrate
        self.equilibration(pinned_monomers=[0])

        if do_plot:
            self.plot_simulation("translocation", "translocation_equilibrated.png")

        # release the monomers and let polymer translocate through the pore
        # track the translocation time
        translocation_t = 0
        # track translocation status
        status = "running"
        # track times in which monomers manage to exit the pore
        exit_times = np.full(self.N, -1.0)
        while  status == "running":
            self.MC_step()
            translocation_t += 1

            exited = self.monomers[:, 0] > self.pore_exit
            not_recorded = exit_times < 0
            to_update = exited & not_recorded
            exit_times[to_update] = translocation_t

            status = self.translocation_stopping_condition()

            if translocation_t % 500 == 0 and do_plot:
                self.plot_simulation("translocation", f"translocation_{translocation_t}.png")

        if do_plot:
            self.plot_simulation("translocation", "translocation_end.png")

        if status == "failure":
            return -1.0
        else:
            waiting_times = np.concatenate(([exit_times[0]], np.diff(exit_times)))
            return translocation_t, waiting_times

    # ------------------------------------------------------------------------------------------------------
    # Plot the simulation...
    # ------------------------------------------------------------------------------------------------------

    def plot_simulation(self, process, filename):
        """
        Plots simulation snapshots.
        :param process: "translocation" or "escape"
        :param filename: name of the file to which snapshot is saved
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # draw wall:
        # - upper half
        wall_up = patches.Rectangle((self.pore_entrance, self.pore_ceil),
                                    width=self.L, height=self.wall_extent,
                                    facecolor='lightblue', edgecolor='blue',
                                    label='Membrane')
        ax.add_patch(wall_up)
        # - lower half
        wall_down = patches.Rectangle((self.pore_entrance, -self.wall_extent),
                                      width=self.L, height=self.wall_extent + self.pore_floor,
                                      facecolor='lightblue', edgecolor='blue',
                                      label='Membrane')
        ax.add_patch(wall_down)

        # plot polymer:
        x = self.monomers[:, 0]
        y = self.monomers[:, 1]
        # - bonds
        ax.plot(x, y, color='black', alpha=0.6, linewidth=1)
        # - monomers
        ax.scatter(x, y, c=np.arange(self.N), cmap='viridis_r', edgecolor='black', s=30)

        # arrange final plot:
        ax.set_aspect('equal')
        ax.set_xlim(-2.5 * self.N, 2.5 * self.N + self.L)
        ax.set_ylim(-2.5 * self.N, 2.5 * self.N + self.w)
        ax.set_title(f"Polymer {process}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if filename is None:
            filename = f"FB_{process}.png"
        plt.savefig(filename)

        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------
    # Animate the simulation...
    # ------------------------------------------------------------------------------------------------------

    def animate_escape(self, filename="escape.gif"):
        """
        Creates a gif of the polymer escape simulation.
        :param filename: name of the file to which gif is saved
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # wall setup:
        # - upper half
        wall_up = patches.Rectangle((self.pore_entrance, self.pore_ceil),
                                    width=self.L, height=self.wall_extent,
                                    color='lightblue', label='Membrane')
        ax.add_patch(wall_up)
        # - lower half
        wall_down = patches.Rectangle((self.pore_entrance, -self.wall_extent),
                                      width=self.L, height=self.wall_extent + self.pore_floor,
                                      color='lightblue', label='Membrane')
        ax.add_patch(wall_down)

        # polymer setup:
        # - bonds
        line, = ax.plot([], [], color='black', alpha=0.6, linewidth=1)
        # - monomers
        dots = ax.scatter([], [], c=[], cmap='viridis_r', vmin=0, vmax=self.N, edgecolor='black', s=30)

        # text overlay setup
        state_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

        # create simple linear monomer with center in the middle of the pore
        self.monomers = np.zeros((self.N, 2))
        spacings = np.arange(self.N) - (self.N - 1) / 2
        bond_length = 3
        self.monomers[:, 0] = bond_length * spacings

        # initialize graphics
        def init():
            ax.set_aspect('equal')
            ax.set_xlim(-3 * self.N, 3 * self.N + self.L)
            ax.set_ylim(-3 * self.N, 3 * self.N + self.w)
            return line, dots

        # track frames for different phases of the process
        def frame_generator():
            # Phase 1: Equilibration (50 frames, Rouse time)
            for f in range(50):
                yield ('Equilibration', f)

            # Phase 2: Translocation
            count = 0
            while not self.escape_stopping_condition():
                yield ('Translocation', count)
                count += 1
                # safety break
                if count > 5000: break

            # Phase 3: Escape complete
            for f in range(10):
                yield ('Done', f)

        # update image for every frame
        def update(gen_data):
            phase, frame_num = gen_data

            # Phase 1: Equilibration (50 frames, Rouse time)
            if phase == 'Equilibration':
                state_text.set_text('Equilibration')

                self.is_equilibration = True

                # pin monomers inside the pore
                pinned_monomers = [m for m in range(self.N)
                                   if self.pore_entrance <= self.monomers[m, 0] <= self.pore_exit]

                # run this step for 50 frames
                steps_per_frame = int((20 * self.N**2.5) / 50)
                for _ in range(steps_per_frame):
                    self.MC_step(pinned_monomers=pinned_monomers)

            # Phase 2: Translocation
            elif phase == 'Translocation':
                state_text.set_text('Translocation (escape)')

                self.is_translocation = False

                for _ in range(20):
                    if not self.escape_stopping_condition():
                            self.MC_step()

            # Phase 3: Escape complete
            else:
                state_text.set_text('Escape Completed')

            # plot
            x = self.monomers[:, 0]
            y = self.monomers[:, 1]
            line.set_data(x, y)
            dots.set_offsets(self.monomers)
            dots.set_array(np.arange(self.N))

            return line, dots, state_text

        # forward to animation function and save the gif
        anim = FuncAnimation(fig, update, frames=frame_generator, init_func=init, blit=True, save_count=1000)
        anim.save(filename, writer=PillowWriter(fps=20))
        plt.close()
        print("Animation saved")

    def animate_translocation(self, filename="translocation.gif"):
        """
        Creates a gif of the polymer escape simulation.
        :param filename: name of the file to which gif is saved
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # wall setup:
        # - upper half
        wall_up = patches.Rectangle((self.pore_entrance, self.pore_ceil),
                                    width=self.L, height=self.wall_extent,
                                    color='lightblue', label='Membrane')
        ax.add_patch(wall_up)
        # - lower half
        wall_down = patches.Rectangle((self.pore_entrance, -self.wall_extent),
                                      width=self.L, height=self.wall_extent + self.pore_floor,
                                      color='lightblue', label='Membrane')
        ax.add_patch(wall_down)

        # polymer setup:
        # - bonds
        line, = ax.plot([], [], color='black', alpha=0.6, linewidth=1)
        # - monomers
        dots = ax.scatter([], [], c=[], cmap='viridis_r', vmin=0, vmax=self.N, edgecolor='black', s=30)

        # text overlay setup
        state_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

        # create simple linear monomer with center in the middle of the pore
        self.monomers = np.zeros((self.N, 2))
        self.monomers[:, 0] = self.pore_entrance - np.arange(self.N)

        # initialize graphics
        def init():
            ax.set_aspect('equal')
            ax.set_xlim(-3 * self.N, 3 * self.N)
            ax.set_ylim(-3 * self.N, 3 * self.N)
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
            for f in range(10):
                yield ('Done', f)

        # update image for every frame
        def update(gen_data):
            phase, frame_num = gen_data

            # Phase 1: Equilibration (50 frames, Rouse time)
            if phase == 'Equilibration':
                state_text.set_text('Equilibration')

                self.is_equilibration = True

                # run this step for 50 frames - only the first monomer is pinned
                steps_per_frame = int((20 * self.N**2.5) / 50)
                for _ in range(steps_per_frame):
                    self.MC_step(pinned_monomers=self.monomers[0])

            # Phase 2: Translocation
            elif phase == 'Translocation':
                state_text.set_text('Translocation (escape)')

                self.is_equilibration = False

                for _ in range(20):
                    if self.translocation_stopping_condition() == "running":
                            self.MC_step()

            # Phase 3: Escape complete
            else:
                final_status = self.translocation_stopping_condition()
                if final_status == "success":
                    state_text.set_text('Translocation Completed Successfully')
                else:
                    state_text.set_text('Translocation Failed')

            # plot
            x = self.monomers[:, 0]
            y = self.monomers[:, 1]
            line.set_data(x, y)
            dots.set_offsets(self.monomers)
            dots.set_array(np.arange(self.N))

            return line, dots, state_text

        # forward to animation function and save the gif
        anim = FuncAnimation(fig, update, frames=frame_generator, init_func=init, blit=True, save_count=1000)
        anim.save(filename, writer=PillowWriter(fps=20))
        plt.close()
        print("Animation saved")
