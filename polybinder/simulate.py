from itertools import combinations_with_replacement as combo
import operator
import os

import gsd.hoomd
import hoomd
import hoomd.md
from mbuild.formats.hoomd_forcefield import create_hoomd_forcefield
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
    wall_time_limit : int, default None
        Set a maximum amount of time in seconds a simulation is allowed
        to run even if it hasn't ran to completion.
    
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
        nlist="Cell",
        dt=0.0001,
        auto_scale=True,
        ref_values=None,
        mode="gpu",
        gsd_write=1e4,
        log_write=1e3,
        seed=42,
        cg_potentials_dir=None,
        restart=None,
        wall_time_limit=None
    ):
        self.r_cut = r_cut
        self.tau_kt = tau_kt
        self.tau_p = tau_p
        self.nlist = getattr(hoomd.md.nlist, nlist)
        self.dt = dt
        self.auto_scale = auto_scale
        self.ref_values = ref_values
        self.mode = mode
        self.gsd_write = gsd_write
        self.log_write = log_write
        self.seed = seed
        self.restart = restart
        self.wall_time_limit = wall_time_limit
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
            "kinetic_temperature",
            "potential_energy",
            "kinetic_energy",
            "volume",
            "pressure",
            "pressure_tensor",
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
        and/or pressure.

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
            Create LJ wall potentials along the specified axis
            of the simulation volume.
            Not compatible with NPT simulations; pressure must be None

        """
        if wall_axis and pressure:
            raise ValueError(
                    "Wall potentials can only be used with the NVT ensemble."
            )
        if [shrink_kT, shrink_steps, shrink_period].count(None) %3 != 0:
            raise ValueError(
                "If shrinking, all of  shrink_kT, shrink_steps and "
                "shrink_period need to be given."
            )
        if shrink_steps is None:
            shrink_steps = 0
        
        if self.cg_system is False:
            init_snap, forcefields, refs = create_hoomd_forcefield(
                    self.system,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
            )
        else:
            #TODO: See what needs to be changed and returned
            #in _create_hoomd_sim..
            init_snap, objs = self._create_hoomd_sim_from_snapshot()

        init_x = init_snap.configuration.box[0]
        init_y = init_snap.configuration.box[1]
        init_z = init_snap.configuration.box[2]
        #TODO: Do I need to set "1-4" here, or is it set in mBuild?
        forcefields[0].nlist.exclusions = ["bond", "1-3"]
        # Create Hoomd simulation object and initialize a state
        #TODO: Change neighbor list from cell to tree if needed
        device = hoomd.device.auto_select()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        if self.restart:
            sim.create_state_from_gsd(self.restart)
        else:
            sim.create_state_from_snapshot(init_snap)
        _all = hoomd.filter.All()
        gsd_writer, table_file, = self._hoomd_writers(
                group=_all, sim=sim, forcefields=forcefields
        )
        sim.operations.writers.append(gsd_writer)
        sim.operations.writers.append(table_file)
        
        if wall_axis is not None: # Set up wall potentials
            wall_force, walls, normal_vector = self._hoomd_walls(
                    wall_axis, init_x, init_y, init_z
            )
            wall_force.params[init_snap.particles.types] = {
                    "epsilon": 1.0,
                    "sigma": 1.0,
                    "r_cut": 2.5,
                    "r_extrap": 0
            }
            forcefields.append(wall_force)

        if shrink_kT and shrink_steps: # Set up shrinking run
            integrator = hoomd.md.Integrator(dt=self.dt)
            integrator.forces = forcefields
            integrator_method = hoomd.md.methods.NVT(
                    filter=_all, kT=shrink_kT, tau=self.tau_kt
            )
            integrator.methods = [integrator_method]
            sim.operations.add(integrator)
            sim.state.thermalize_particle_momenta(filter=_all, kT=shrink_kT)
            box_resize_trigger = hoomd.trigger.Periodic(shrink_period)
            ramp = hoomd.variant.Ramp(
                A=0, B=1, t_start=sim.timestep, t_ramp=int(shrink_steps)
            ) 
            #TODO: Add the box stuff to its own function?
            initial_box = sim.state.box
            final_box = hoomd.Box(
                    Lx=self.target_box[0],
                    Ly=self.target_box[1],
                    Lz=self.target_box[2]
            )
            box_resize = hoomd.update.BoxResize(
                    box1=initial_box,
                    box2=final_box,
                    variant=ramp,
                    trigger=box_resize_trigger
            )
            sim.operations.updaters.append(box_resize)

            if wall_axis is not None:
                pass
                # TODO: Update walls during shrink?
            else: # Run shrink steps without updating walls
                sim.run(shrink_steps + 1)
            assert sim.state.box == final_box

        if pressure: # Set NPT integrator
            try: 
                integrator # Not yet defined if no shrink step ran
                sim.operations.remove(integrator)
            except NameError:
                integrator = hoomd.md.Integrator(dt=self.dt)
                integrator.forces = forcefields
            integrator_method = hoomd.md.methods.NPT(
                    filter=_all, kT=kT, S=pressure, tauS=self.tau_p
            )
            integrator.methods = [integrator_method]
            sim.operations.add(integrator)
        else: # Set NVT integrator 
            try: 
                integrator # Not yet defined if no shrink step ran
                sim.operations.remove(integrator)
            except NameError:
                integrator = hoomd.md.Integrator(dt=self.dt)
                integrator.forces = forcefields
            integrator_method = hoomd.md.methods.NVT(
                    filter=_all, kT=kT, tau=self.tau_kt
            )
            integrator.methods = [integrator_method]
            sim.operations.add(integrator)

        sim.state.thermalize_particle_momenta(filter=_all, kT=kT)
        try:
            while sim.timestep < n_steps + shrink_steps + 1:
                #TODO: Use a better approach here avoid an odd amount of steps?
                sim.run(n_steps + shrink_steps + 1)
                #sim.run(min(10000, n_steps + shrink_steps + 1 - sim.timestep))
                if self.wall_time_limit:
                    if (sim.device.communicator.walltime + sim.walltime >=
                            self.wall_time_limit):
                        break
        finally:
            hoomd.write.GSD.write(
                    state=sim.state, mode='wb', filename="restart.gsd"
            )
		
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

        if self.cg_system is False:
            init_snap, forcefields, refs = create_hoomd_forcefield(
                    self.system,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
            )
        else:
            #TODO: See what needs to be changed and returned
            #in _create_hoomd_sim..
            init_snap, objs = self._create_hoomd_sim_from_snapshot()

        init_x = init_snap.configuration.box[0]
        init_y = init_snap.configuration.box[1]
        init_z = init_snap.configuration.box[2]
        #TODO: Do I need to set "1-4" here, or is it set in mBuild?
        forcefields[0].nlist.exclusions = ["bond", "1-3"]
        # Create Hoomd simulation object and initialize a state
        device = hoomd.device.auto_select()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        #TODO: Change nlist from cell to tree if needed
        if self.restart:
            sim.create_state_from_gsd(self.restart)
        else:
            sim.create_state_from_snapshot(init_snap)
        _all = hoomd.filter.All()
        gsd_writer, table_file = self._hoomd_writers(
                group=_all, sim=sim, forcefields=forcefields
        )
        sim.operations.writers.append(gsd_writer)
        #sim.operations.writers.append(table_file)
        
        if wall_axis is not None: # Set up wall potentials
            wall_force, walls, normal_vector = self._hoomd_walls(
                    wall_axis, init_x, init_y, init_z
            )
            wall_force.params[init_snap.particles.types] = {
                    "epsilon": 1.0,
                    "sigma": 1.0,
                    "r_cut": 2.5,
                    "r_extrap": 0
            }
            forcefields.append(wall_force)

        if shrink_kT and shrink_steps: # Set up shrinking run
            integrator = hoomd.md.Integrator(dt=self.dt)
            integrator.forces = forcefields
            integrator_method = hoomd.md.methods.NVT(
                    filter=_all, kT=shrink_kT, tau=self.tau_kt
            )
            integrator.methods = [integrator_method]
            sim.operations.add(integrator)
            sim.state.thermalize_particle_momenta(filter=_all, kT=shrink_kT)
            box_resize_trigger = hoomd.trigger.Periodic(shrink_period)
            ramp = hoomd.variant.Ramp(
                A=0, B=1, t_start=sim.timestep, t_ramp=int(shrink_steps)
            ) 
            #TODO: Add the box stuff to its own function?
            initial_box = sim.state.box
            final_box = hoomd.Box(
                    Lx=self.target_box[0],
                    Ly=self.target_box[1],
                    Lz=self.target_box[2]
            )
            box_resize = hoomd.update.BoxResize(
                    box1=initial_box,
                    box2=final_box,
                    variant=ramp,
                    trigger=box_resize_trigger
            )
            sim.operations.updaters.append(box_resize)

            if wall_axis is not None:
                pass
                # TODO: Update walls during shrink?
            else: # Run shrink steps without updating walls
                sim.run(shrink_steps + 1)
            assert sim.state.box == final_box

        if pressure: # Set NPT integrator
            try: 
                integrator # Not yet defined if no shrink step ran
                sim.operations.remove(integrator)
            except NameError:
                integrator = hoomd.md.Integrator(dt=self.dt)
                integrator.forces = forcefields
            integrator_method = hoomd.md.methods.NPT(
                    filter=_all, kT=kT, S=pressure, tauS=self.tau_p
            )
            integrator.methods = [integrator_method]
            sim.operations.add(integrator)
        else: # Set NVT integrator 
            try: 
                integrator # Not yet defined if no shrink step ran
                sim.operations.remove(integrator)
            except NameError:
                integrator = hoomd.md.Integrator(dt=self.dt)
                integrator.forces = forcefields

        last_step = shrink_steps
        for kT in schedule:
            if pressure:
                integrator_method = hoomd.md.methods.NPT(
                        filter=_all, kT=kT, S=pressure, tauS=self.tau_p
                )
                integrator.methods = [integrator_method]
            else:
                integrator_method = hoomd.md.methods.NVT(
                        filter=_all, kT=kT, tau=self.tau_kt
                )
                integrator.methods = [integrator_method]

            sim.operations.add(integrator)
            sim.state.thermalize_particle_momenta(filter=_all, kT=kT)
            n_steps = schedule[kT]
            sim.run(n_steps) 
            sim.operations.remove(integrator)

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
        if self.cg_system is False:
            init_snap, forcefields, refs = create_hoomd_forcefield(
                    self.system,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
            )
        else:
            #TODO: See what needs to be changed and returned
            #in _create_hoomd_sim..
            init_snap, objs = self._create_hoomd_sim_from_snapshot()

        device = hoomd.device.auto_select()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        if self.restart:
            sim.create_state_from_gsd(self.restart)
        else:
            sim.create_state_from_snapshot(init_snap)
        _all = hoomd.filter.All()
        gsd_writer, table_file = self._hoomd_writers(
                group=_all, sim=sim, forcefields=forcefields
        )
        sim.operations.writers.append(gsd_writer)
        #sim.operations.writers.append(table_file)
        init_box = sim.state.box

        tensile_axis = tensile_axis.lower()
        # TODO: Call on the init_box here instead
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




        ## BEGIN HOOMD 2 STUFF:
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
    
    def _hoomd_writers(self, group, forcefields, sim):
        # GSD and Logging:
        if self.restart:
            writemode = "a"
        else:
            writemode = "w"
        gsd_writer = hoomd.write.GSD(
                filename="sim_traj.gsd",
                trigger=hoomd.trigger.Periodic(int(self.gsd_write)),
                mode=f"{writemode}b",
                dynamic=["momentum"]
        )

        logger = hoomd.logging.Logger(categories=["scalar", "string"])
        logger.add(sim, quantities=["timestep", "tps"])
        thermo_props = hoomd.md.compute.ThermodynamicQuantities(filter=group)
        sim.operations.computes.append(thermo_props)
        logger.add(thermo_props, quantities=self.log_quantities)
        for f in forcefields:
            logger.add(f, quantities=["energy"])

        table_file = hoomd.write.Table(
            output=open("sim_traj.txt", mode=f"{writemode}", newline="\n"),
            trigger=hoomd.trigger.Periodic(period=int(self.log_write)),
            logger=logger,
            max_header_len=None,
        )
        return gsd_writer, table_file 

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
                hoomd_system,
                self.nlist(),
                pair_pot,
                bond_pot,
                angle_pot,
        ]
        return init_snap, hoomd_objs 

    def _hoomd_walls(self, wall_axis, Lx, Ly, Lz):
        """Create hoomd LJ wall potentials"""
        wall_origin = np.asarray(wall_axis) * np.array(
                [Lx/2, Ly/2, Lz/2]
        )
        normal_vector = -np.asarray(wall_axis)
        wall_origin2 = -wall_origin
        normal_vector2 = -normal_vector
        wall1 = hoomd.wall.Plane(origin=wall_origin, normal=normval_vector)
        wall2 = hoomd.wall.Plane(origin=wall_origin2, normal=normal_vector2)
        walls = [wall1, wall2]
        lj_wall = hoomd.md.external.wall.LJ(
                walls=walls, default_r_cut=2.5
        )
        return lj_wall, walls, normal_vector
