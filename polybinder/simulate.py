from itertools import combinations_with_replacement as combo
import operator
import os

import gsd.hoomd
import hoomd
import hoomd.md
from hoomd.md import wall
from mbuild.formats.hoomd_simulation import create_hoomd_simulation
import numpy as np
import parmed as pmd

from polybinder.library import COMPOUND_DIR, SYSTEM_DIR, FF_DIR


class Simulation:
    """The simulation context management class.

    This class takes the output of the Initialization class
    and sets up a hoomd-blue simulation.

    Parameters
    ----------
    system : system.Initializer
        The system created in polybinder.system
    r_cut : float, default 2.5
        Cutoff radius for potentials (in simulation distance units)
    tau_kt : float, default 0.1
        Thermostat coupling period (in simulation time units)
    tau_p : float, default None
        Barostat coupling period (in simulation time units)
    nlist : str, default `cell`
        Type of neighborlist to use. Options are "cell", "tree", and "stencil".
        See https://hoomd-blue.readthedocs.io/en/stable/nlist.html and
        https://hoomd-blue.readthedocs.io/en/stable/module-md-nlist.html
    dt : float, default 0.0001
        Size of simulation timestep (in simulation time units)
    auto_scale : bool, default True
        Set to true to use reduced simulation units.
        distance, mass, and energy are scaled by the largest value
        present in the system for each.
    ref_values : dict, default None
        Define the reference units for distance, mass, energy.
        Set auto_scale to False to define your own reference values.
    mode : str, default "gpu"
        Mode flag passed to hoomd.context.initialize. Options are "cpu" and
        "gpu".
    gsd_write : int, default 1e4
        Period to write simulation snapshots to gsd file.
    log_write : int, default 1e3
        Period to write simulation data to the log file.
    seed : int, default 42
        Seed passed to integrator when randomizing velocities.
    cg_potentials_dir : str, default None
        Directory inside of polybinder.library.forcefields to
        look for coarse-grained system potentials. If left
        as None, then it will only look in polybinder.library.forcefields.
        This is only used when `system` has been coarse-grained in
        polybinder.system
    restart : str, default None
        Path to gsd file from which to restart the simulation
    
    Methods
    -------
    quench: Runs a hoomd simulation
        Run a simulation at a single temperature in NVT or a single
        temperature and pressure in NPT
    anneal: Runs a hoomd simulation
        Define a schedule of temperature and steps to follow over the
        course of the simulation. Can be used in NVT or NPT at a single
        pressure.

    """
    def __init__(
        self,
        system,
        r_cut=2.5,
        tau_kt=0.1,
        tau_p=None,
        nlist="cell",
        dt=0.0001,
        auto_scale=True,
        ref_values=None,
        mode="gpu",
        gsd_write=1e4,
        log_write=1e3,
        seed=42,
        cg_potentials_dir=None,
        restart=None
    ):
        self.r_cut = r_cut
        self.tau_kt = tau_kt
        self.tau_p = tau_p
        self.nlist = getattr(hoomd.md.nlist, nlist.lower())
        self.dt = dt
        self.auto_scale = auto_scale
        self.ref_values = ref_values
        self.mode = mode
        self.gsd_write = gsd_write
        self.log_write = log_write
        self.seed = seed
        self.restart = restart
        # Coarsed-grained related parameters, system is a str (file path of GSD)
        if isinstance(system.system, str):
            assert ref_values != None, (
                        "Autoscaling is not supported for coarse-grain sims."
                        "Provide the relevant reference units"
                    )

            self.system = system.system
            self.cg_system = True
            if cg_potentials_dir is None:
                self.cg_ff_path = FF_DIR
            else:
                self.cg_ff_path = f"{FF_DIR}/{cg_potentials_dir}"
            self.ref_energy = ref_values["energy"]
            self.ref_distance = ref_values["distance"]
            self.ref_mass = ref_values["mass"]
        # Non coarse-grained related parameters, system is a pmd.Structure 
        elif isinstance(system.system, pmd.Structure):
            self.system = system.system
            self.cg_system = False
            if ref_values and not auto_scale:
                self.ref_energy = ref_values["energy"]
                self.ref_distance = ref_values["distance"]
                self.ref_mass = ref_values["mass"]
            # Pulled from mBuild hoomd_simulation.py
            elif auto_scale and not ref_values:
                self.ref_mass = max([atom.mass for atom in self.system.atoms])
                pair_coeffs = list(
                set(
                    (atom.type, atom.epsilon, atom.sigma)
                    for atom in self.system.atoms
                )
            )
                self.ref_energy = max(pair_coeffs, key=operator.itemgetter(1))[1]
                self.ref_distance = max(pair_coeffs, key=operator.itemgetter(2))[2]

        if system.system_type != "interface":
            # Conv from nm (mBuild) to ang (parmed) and set to reduced length 
            self.target_box = system.target_box * 10 / self.ref_distance

        self.log_quantities = [
            "temperature",
            "pressure",
            "pressure_xx",
            "pressure_yy",
            "pressure_zz",
            "pressure_xy",
            "pressure_xz",
            "pressure_yz",
            "volume",
            "potential_energy",
            "kinetic_energy",
            "pair_lj_energy",
            "bond_harmonic_energy",
            "angle_harmonic_energy",
        ]

    def quench(
        self,
        n_steps,
        kT=None,
        pressure=None,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
        wall_axis=None,
        **kwargs
    ):
        """Runs an NVT or NPT simulation at a single temperature
        and pressure.

        Call this funciton after initializing the Simulation class.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to run the simulation.
        kT : float, default None
            The dimensionless temperature at which to run the simulation
        pressure : float, default None
            The dimensionless pressure at which to run the simulation
        shrink_kT : float, default None
            The dimensionless temperature to use during the shrink steps
        shrink_steps : int, defualt None
            The number of steps to run during the shrink process
        shrink_period : int, default None
            The period between box updates during shrinking
        wall_axis : (1,3) array like, default None
            Create LJ wall potentials along the specified axis of the simulation volume.
            Not compatible with NPT simulations; pressure must be None

        """
        if wall_axis and pressure:
            raise ValueError(
                    "Wall potentials can only be used with the NVT ensemble."
                    )
        if [shrink_kT, shrink_steps, shrink_period].count(None) %3 != 0:
            raise ValueError(
                "If shrinking, all of  shrink_kT, shrink_steps and "
                "shrink_periopd need to be given."
            )
        if shrink_steps is None:
            shrink_steps = 0

        hoomd_args = f"--single-mpi --mode={self.mode}"
        sim = hoomd.context.initialize(hoomd_args)
        with sim:
            if self.cg_system is False:
                objs, refs = create_hoomd_simulation(
                    self.system,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
                    nlist=self.nlist,
                    restart=self.restart
                )
                if self.restart is None:
                    init_x = objs[0].box.Lx
                    init_y = objs[0].box.Ly
                    init_z = objs[0].box.Lz
                else:
                    init_x = objs[0].configuration.box[0]
                    init_y = objs[0].configuration.box[1]
                    init_z = objs[0].configuration.box[2]

            elif self.cg_system is True:
                objs = self._create_hoomd_sim_from_snapshot()
                self.log_quantities.remove("pair_lj_energy")
                init_x = objs[0].configuration.box[0]
                init_y = objs[0].configuration.box[1]
                init_z = objs[0].configuration.box[2]

            init_snap = objs[0]
            hoomd_system = objs[1]
            _all = hoomd.group.all()
            hoomd.md.integrate.mode_standard(dt=self.dt)
            hoomd.dump.gsd(
                "sim_traj.gsd",
                period=self.gsd_write,
                group=_all,
                phase=0,
                dynamic=["momentum"],
                overwrite=False,
            )
            hoomd.analyze.log(
                "sim_traj.log",
                period=self.log_write,
                quantities=self.log_quantities,
                header_prefix="#",
                overwrite=True,
                phase=0,
            )

            if wall_axis is not None:
                wall_force, walls, normal_vector = self._hoomd_walls(
                        wall_axis, init_x, init_y, init_z
                )
                wall_force.force_coeff.set(
                    init_snap.particles.types,
                    sigma=1.0,
                    epsilon=1.0,
                    r_extrap=0
                )

            if shrink_kT and shrink_steps:
                integrator = hoomd.md.integrate.nvt(
                        group=_all,
                        kT=shrink_kT,
                        tau=self.tau_kt
                )
                integrator.randomize_velocities(seed=self.seed)
                x_variant = hoomd.variant.linear_interp(
                        [(0, init_x), (shrink_steps, self.target_box[0])]
                )
                y_variant = hoomd.variant.linear_interp(
                        [(0, init_y), (shrink_steps, self.target_box[1])]
                )
                z_variant = hoomd.variant.linear_interp(
                        [(0, init_z), (shrink_steps, self.target_box[2])]
                )

                box_updater = hoomd.update.box_resize(
                    Lx=x_variant, 
                    Ly=y_variant,
                    Lz=z_variant,
                    period=shrink_period
                )
                # Update wall origins during shrinking
                momentum = hoomd.md.update.zero_momentum(period=shrink_steps)
                if wall_axis is not None:
                    step = 0
                    while step < shrink_steps:
                        hoomd.run_upto(step + shrink_period)
                        current_box = np.array([
                                hoomd_system.box.Lx,
                                hoomd_system.box.Ly,
                                hoomd_system.box.Lz
                        ])
                        walls.del_plane([0, 1])
                        walls.add_plane(
                                (current_box/2 * wall_axis), normal_vector
                        )
                        walls.add_plane(
                                (-current_box/2 * wall_axis), -normal_vector
                        )
                        step += shrink_period
                else:
                    hoomd.run_upto(shrink_steps)

                box_updater.disable()
                momentum.disable()

            gsd_restart = hoomd.dump.gsd(
                "restart.gsd",
                period=self.gsd_write,
                group=_all,
                truncate=True,
                phase=0,
                dynamic=["momentum"]
            )
            # Run the primary simulation
            if pressure:
                try: # Not defined if no shrink step
                    integrator.disable() 
                except NameError:
                    pass
                integrator = hoomd.md.integrate.npt(
                        group=_all,
                        tau=self.tau_kt,
                        tauP=self.tau_p,
                        P=pressure,
                        kT=kT
                        )
            elif not pressure:
                try: # Integrator already created (shrinking), update kT
                    integrator.set_params(kT=kT) 
                except NameError: # Integrator not yet created (no shrinking)
                    integrator = hoomd.md.integrate.nvt(
                            group=_all,
                            tau=self.tau_kt,
                            kT=kT)
            integrator.randomize_velocities(seed=self.seed)
            try:
                hoomd.run_upto(n_steps + shrink_steps)
                print("Simulation completed")
                done=True
            except hoomd.WalltimeLimitReached:
                print("Walltime limit reached")
                done=False
            finally:
                gsd_restart.write_restart()
                print("Restart GSD file written")
        return done

    def anneal(
        self,
        kT_init=None,
        kT_final=None,
        pressure=None,
        step_sequence=None,
        schedule=None,
        wall_axis=None,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
    ):
        if wall_axis and pressure:
            raise ValueError(
                "Wall potentials can only be used with the NVT ensemble"
            )
        if [shrink_kT, shrink_steps, shrink_period].count(None) %3 != 0:
            raise ValueError(
                "If shrinking, then all of shirnk_kT, shrink_steps "
                "and shrink_period need to be given"
            )
        if shrink_steps is None:
            shrink_steps = 0

        if not schedule:
            temps = np.linspace(kT_init, kT_final, len(step_sequence))
            temps = [np.round(t, 1) for t in temps]
            schedule = dict(zip(temps, step_sequence))

        # Get hoomd stuff set:
        hoomd_args = f"--single-mpi --mode={self.mode}"
        sim = hoomd.context.initialize(hoomd_args)
        with sim:
            if self.cg_system is False:
                objs, refs = create_hoomd_simulation(
                    self.system,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
                    nlist=self.nlist
                )
                init_x = objs[0].box.Lx
                init_y = objs[0].box.Ly
                init_z = objs[0].box.Lz
            elif self.cg_system is True:
                objs = self._create_hoomd_sim_from_snapshot()
                self.log_quantities.remove("pair_lj_energy")
                init_x = objs[0].configuration.box[0]
                init_y = objs[0].configuration.box[1]
                init_z = objs[0].configuration.box[2]

            hoomd_system = objs[1]
            init_snap = objs[0]
            _all = hoomd.group.all()
            hoomd.md.integrate.mode_standard(dt=self.dt)
            hoomd.dump.gsd(
                "sim_traj.gsd",
                period=self.gsd_write,
                group=_all,
                phase=0,
                dynamic=["momentum"],
                overwrite=False,
            )
            hoomd.analyze.log(
                "sim_traj.log",
                period=self.log_write,
                quantities=self.log_quantities,
                header_prefix="#",
                overwrite=True,
                phase=0,
            )
            # Set up wall LJ potentials
            if wall_axis is not None:
                wall_force, walls, normal_vector = self._hoomd_walls(
                        wall_axis, init_x, init_y, init_z
                )
                wall_force.force_coeff.set(
                    init_snap.particles.types,
                    sigma=1.0,
                    epsilon=1.0,
                    r_extrap=0
                )

            if shrink_kT and shrink_steps:
                integrator = hoomd.md.integrate.nvt(
                        group=_all,
                        tau=self.tau_kt,
                        kT=shrink_kT
                )
                integrator.randomize_velocities(seed=self.seed)
                x_variant = hoomd.variant.linear_interp(
                        [(0, init_x), (shrink_steps, self.target_box[0])]
                )
                y_variant = hoomd.variant.linear_interp(
                        [(0, init_y), (shrink_steps, self.target_box[1])]
                )
                z_variant = hoomd.variant.linear_interp(
                        [(0, init_z), (shrink_steps, self.target_box[2])]
                )

                box_updater = hoomd.update.box_resize(
                    Lx=x_variant,
                    Ly=y_variant,
                    Lz=z_variant,
                    period=shrink_period
                )
                # Update walls due to shrink box changes
                if wall_axis is not None:
                    step = 0
                    while step < shrink_steps:
                        hoomd.run_upto(step + shrink_period)
                        current_box = np.array([
                                hoomd_system.box.Lx,
                                hoomd_system.box.Ly,
                                hoomd_system.box.Lz
                        ])
                        walls.del_plane([0, 1])
                        walls.add_plane(
                                (current_box/2 * wall_axis), normal_vector
                        )
                        walls.add_plane(
                                (-current_box/2 * wall_axis), -normal_vector
                        )
                        step += shrink_period
                else:
                    hoomd.run_upto(shrink_steps)
                box_updater.disable()

            gsd_restart = hoomd.dump.gsd(
                "restart.gsd",
                period=self.gsd_write,
                group=_all,
                truncate=True,
                phase=0,
                dynamic=["momentum"]
            )

            if pressure:
                try:
                    integrator.disable()
                except NameError:
                    pass
                integrator = hoomd.md.integrate.npt(
                        group=_all,
                        tau=self.tau_kt,
                        tauP=self.tau_p,
                        P=pressure,
                        kT=1
                )
            elif not pressure:
                try:
                    integrator
                except NameError:
                    integrator = hoomd.md.integrate.nvt(
                            group=_all,
                            tau=self.tau_kt,
                            kT=1
                    )
            
            last_step = shrink_steps 
            for kT in schedule: 
                n_steps = schedule[kT]
                integrator.set_params(kT=kT)
                integrator.randomize_velocities(seed=self.seed)
                print(f"Running @ Temp = {kT} kT")
                print(f"Running for {n_steps} steps")
                try:
                    hoomd.run_upto(n_steps + last_step)
                    last_step += n_steps
                    done = True
                    last_temp = kT
                    return done, last_temp
                except hoomd.WalltimeLimitReached:
                    done = False
                    last_temp = kT
                    return done, last_temp
                finally:
                    gsd_restart.write_restart()

    def tensile(self,
            kT,
            strain,
            n_steps,
            expand_period,
            tensile_axis="x",
            fix_ratio=0.05
    ):
        """Runs a simulation of a tensile test pulling along the x-axis.

        Parameters:
        -----------
        strain : float
            The distance to strain the volume along the x-axis
            It is the percentage of the initial volume's x length.
        n_steps : int
            The number of simulation time steps to run.
        expand_period : int
            The number of steps ran between each box update.
        fix_ratio : float, default = 0.05
            The distance along the x-axis to fix particles in place.
            Treated as a percentage of the initial  volume's x_length.
            Since particles are fixed on each side, half of x_fix
            is used for the distance.

        """
        hoomd_args = f"--single-mpi --mode={self.mode}"
        sim = hoomd.context.initialize(hoomd_args)
        with sim:
            objs, refs = create_hoomd_simulation(
                    self.system,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
                    nlist=self.nlist
            )
            hoomd_system = objs[1]
            init_snap = objs[0]
            tensile_axis = tensile_axis.lower()
            init_length = getattr(init_snap.box, f"L{tensile_axis}")
            fix_length = init_length * fix_ratio
            target_length = init_length * (1+strain)
            linear_variant = hoomd.variant.linear_interp(
                    [(0, init_length), (n_steps, target_length)]
            )
            axis_dict = {
                "x": np.array([1,0,0]),
                "y": np.array([0,1,0]),
                "z": np.array([0,0,1])
            }
            adjust_axis = axis_dict[tensile_axis]

            if tensile_axis == "x":
                fix_left = hoomd.group.cuboid( # Negative x side of box
                        name="left",
                        xmin=-init_snap.box.Lx / 2,
                        xmax=(-init_snap.box.Lx / 2)  + fix_length
                )
                fix_right = hoomd.group.cuboid(
                        name="right",
                        xmin=(init_snap.box.Lx / 2) - fix_length,
                        xmax=init_snap.box.Lx / 2
                )
                box_updater = hoomd.update.box_resize(
                        Lx=linear_variant,
                        period=expand_period,
                        scale_particles=False
                )
            elif tensile_axis == "y":
                fix_left = hoomd.group.cuboid( # Negative x side of box
                        name="left",
                        ymin=-init_snap.box.Ly / 2,
                        ymax=(-init_snap.box.Ly / 2)  + fix_length
                )
                fix_right = hoomd.group.cuboid(
                        name="right",
                        ymin=(init_snap.box.Ly / 2) - fix_length,
                        ymax=init_snap.box.Ly / 2
                )
                box_updater = hoomd.update.box_resize(
                        Ly=linear_variant,
                        period=expand_period,
                        scale_particles=False
                )
            elif tensile_axis == "z":
                fix_left = hoomd.group.cuboid( # Negative x side of box
                        name="left",
                        zmin=-init_snap.box.Lz / 2,
                        zmax=(-init_snap.box.Lz / 2)  + fix_length
                )
                fix_right = hoomd.group.cuboid(
                        name="right",
                        zmin=(init_snap.box.Lz / 2) - fix_length,
                        zmax=init_snap.box.Lz / 2
                )
                box_updater = hoomd.update.box_resize(
                        Lz=linear_variant,
                        period=expand_period,
                        scale_particles=False
                )

            _all_fixed = hoomd.group.union(
                    name="fixed", a=fix_left, b=fix_right
            )
            _all = hoomd.group.all()
            _integrate = hoomd.group.difference(
                    name="integrate", a=_all, b=_all_fixed
            )
            hoomd.md.integrate.mode_standard(dt=self.dt)
            integrator = hoomd.md.integrate.nve(
                    group=_integrate, limit=None, zero_force=False
            )
            integrator.randomize_velocities(kT, seed=self.seed)

            hoomd.dump.gsd(
                    "sim_traj.gsd",
                    period=self.gsd_write,
                    group=_all,
                    phase=0,
                    dynamic=["momentum"],
                    overwrite=False
            )
            hoomd.analyze.log(
                    "sim_traj.log",
                    period=self.log_write,
                    quantities=self.log_quantities,
                    header_prefix="#",
                    overwrite=True,
                    phase=0
            )
            gsd_restart = hoomd.dump.gsd(
                    "restart.gsd",
                    period=self.gsd_write,
                    group=_all,
                    truncate=True,
                    phase=0,
                    dynamic=["momentum"]
            )
            
            # Start simulation run
            adjust_axis = axis_dict[tensile_axis]
            step = 0
            last_L = init_length 
            while step < n_steps:
                try:
                    hoomd.run_upto(step + expand_period)
                    current_L = getattr(hoomd_system.box, f"L{tensile_axis}")
                    diff = current_L - last_L
                    for particle in fix_left:
                        particle.position -= (adjust_axis * (diff/2))
                    for particle in fix_right:
                        particle.position += (adjust_axis * (diff/2))
                    step += expand_period
                    last_L = current_L 
                except hoomd.WalltimeLimitReached:
                    pass
                finally:
                    gsd_restart.write_restart()

    def _create_hoomd_sim_from_snapshot(self):
        """Creates needed hoomd objects.

        Similar to the `create_hoomd_simulation` function
        from mbuild, but designed to work when initializing
        a system from a gsd file rather than a Parmed structure.
        Created specifically for using table potentials with
        coarse-grained systems.

        """
        if self.restart is None:
            hoomd_system = hoomd.init.read_gsd(self.system)
            with gsd.hoomd.open(self.system, "rb") as f:
                init_snap = f[0]
        else:
            with gsd.hoomd.open(self.restart) as f:
                init_snap = f[-1]
                hoomd_system = hoomd.init.read_gsd(
                    self.restart, restart=self.restart
                )
                print("Simulation initialized from restart file")

        pairs = []
        pair_pot_files = []
        pair_pot_widths = []
        for pair in [list(i) for i in combo(init_snap.particles.types, r=2)]:
            _pair = "-".join(sorted(pair))
            pair_pot_file = f"{self.cg_ff_path}/{_pair}.txt"
            try:
                assert os.path.exists(pair_pot_file)
            except AssertionError:
                raise RuntimeError(f"The potential file {pair_pot_file} "
                    f"for pair {_pair} does not exist in {self.cg_ff_path}."
                )
            pairs.append(_pair)
            pair_pot_files.append(pair_pot_file)
            pair_pot_widths.append(len(np.loadtxt(pair_pot_file)[:,0]))

        if not all([i == pair_pot_widths[0] for i in pair_pot_widths]):
            raise RuntimeError(
                "All pair potential files must have the same length"    
            )

        pair_pot = hoomd.md.pair.table(
                width=pair_pot_widths[0], nlist=self.nlist()
        )
        for pair, fpath in zip(pairs, pair_pot_files):
            pair = pair.split("-")
            pair_pot.set_from_file(f"{pair[0]}", f"{pair[1]}", filename=fpath)

        # Repeat same process for Bonds 
        bonds = []
        bond_pot_files = []
        bond_pot_widths = []
        for bond in init_snap.bonds.types:
            fname = f"{bond}_bond.txt"
            bond_pot_file = f"{self.cg_ff_path}/{fname}"
            try:
                assert os.path.exists(bond_pot_file)
            except AssertionError:
                raise RuntimeError(f"The potential file {bond_pot_file} "
                    f"for bond {bond} does not exist in {self.cg_ff_path}."
                )
            bonds.append(bond)
            bond_pot_files.append(bond_pot_file)
            bond_pot_widths.append(len(np.loadtxt(bond_pot_file)[:,0]))

        if not all([i == bond_pot_widths[0] for i in bond_pot_widths]):
            raise RuntimeError(
                "All bond potential files must have the same length"    
            )

        bond_pot = hoomd.md.bond.table(width=bond_pot_widths[0])
        for bond, fpath in zip(bonds, bond_pot_files):
            bond_pot.set_from_file(f"{bond}", f"{bond_pot_file}")
        
        # Repeat same process for Angles 
        angles = []
        angle_pot_files = []
        angle_pot_widths = []
        for angle in init_snap.angles.types:
            fname = f"{angle}_angle.txt"
            angle_pot_file = f"{self.cg_ff_path}/{fname}"
            try:
                assert os.path.exists(angle_pot_file)
            except AssertionError:
                raise RuntimeError(f"The potential file {angle_pot_file} "
                    f"for angle {angle} does not exist in {self.cg_ff_path}."
                )
            angles.append(angle)
            angle_pot_files.append(angle_pot_file)
            angle_pot_widths.append(len(np.loadtxt(angle_pot_file)[:,0]))

        if not all([i == angle_pot_widths[0] for i in angle_pot_widths]):
            raise RuntimeError(
                "All bond potential files must have the same length"    
            )

        angle_pot = hoomd.md.angle.table(width=angle_pot_widths[0])
        for angle, fpath in zip(angles, angle_pot_files):
            angle_pot.set_from_file(f"{angle}", f"{angle_pot_file}")

        hoomd_objs = [
                init_snap,
                hoomd_system,
                self.nlist(),
                pair_pot,
                bond_pot,
                angle_pot,
        ]
        return hoomd_objs 

    def _hoomd_walls(self, wall_axis, init_x, init_y, init_z):
        """Create hoomd LJ wall potentials"""
        wall_origin = np.asarray(wall_axis) * np.array(
                [init_x/2, init_y/2, init_z/2]
                )
        normal_vector = -np.asarray(wall_axis)
        wall_origin2 = -wall_origin
        normal_vector2 = -normal_vector
        walls = wall.group(
            wall.plane(
                origin=wall_origin, normal=normal_vector, inside=True
                ),
            wall.plane(
                origin=wall_origin2, normal=normal_vector2, inside=True
                ),
        )
        wall_force = wall.lj(walls, r_cut=2.5)
        return wall_force, walls, normal_vector
