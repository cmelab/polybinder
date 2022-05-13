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
    nlist : str, default `Cell`
        Type of neighborlist to use. Options are "Cell", "Tree", and "Stencil".
        See https://hoomd-blue.readthedocs.io/en/latest/module-md-nlist.html
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
    tensile: Runs a hoomd simulation
        Use this simulation method to perform a tensile test on the 
        simulation volume. 

    """
    def __init__(
        self,
        system,
        r_cut=2.5,
        tau_kt=0.1,
        tau_p=None,
        nlist="Cell",
        wall_axis=None,
        dt=0.0003,
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
        self.wall_axis = wall_axis
        self.dt = dt
        self.auto_scale = auto_scale
        self.ref_values = ref_values
        self.mode = mode
        self.gsd_write = gsd_write
        self.log_write = log_write
        self.seed = seed
        self.restart = restart
        self.wall_time_limit = wall_time_limit
        self.system = system.system
        self.ran_shrink = False

        # Coarsed-grained related parameters, system is a str (path to a GSD)
        if isinstance(self.system, str):
            assert ref_values != None, (
                        "Autoscaling is not supported for coarse-grain sims. "
                        "Provide the relevant reference units"
            )
            self.cg_system = True
            if cg_potentials_dir is None:
                self.cg_ff_path = FF_DIR
            else:
                self.cg_ff_path = f"{FF_DIR}/{cg_potentials_dir}"

            self.ref_energy = ref_values["energy"]
            self.ref_distance = ref_values["distance"]
            self.ref_mass = ref_values["mass"]

        # Non coarse-grained related parameters, system is a pmd.Structure 
        elif isinstance(self.system, pmd.Structure):
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

        # Set target volume used during shrinking
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

        self.device = hoomd.device.auto_select()
        self.sim = hoomd.Simulation(device=self.device, seed=self.seed)

        # Initialize the sim state.
        if not self.cg_system:
            self.init_snap, self.forcefields, refs = create_hoomd_forcefield(
                    structure=self.system,
                    r_cut=self.r_cut,
                    ref_distance=self.ref_distance,
                    ref_mass=self.ref_mass,
                    ref_energy=self.ref_energy,
                    auto_scale=self.auto_scale,
            )
            if self.restart:
                self.sim.create_state_from_gsd(self.restart)
            else:
                self.sim.create_state_from_snapshot(self.init_snap)
        else:
            self.init_snap, self.forcefields = self._create_hoomd_sim_from_snapshot()
            self.sim.create_state_from_snapshot(self.init_snap)

        # Set up wall potentials
        if self.wall_axis:
            walls = self._hoomd_walls(
                    Lx = self.init_snap.configuration.box[0],
                    Ly = self.init_snap.configuration.box[1],
                    Lz = self.init_snap.configuration.box[2],
            )
            self.lj_walls = hoomd.md.external.wall.LJ(walls=walls)
            self.lj_walls.params[self.init_snap.particles.types] = {
                    "epsilon": 1.0,
                    "sigma": 1.0,
                    "r_cut": 2.5,
                    "r_extrap": 0
            }
            self.forcefields.append(self.lj_walls)

        # Default nlist is Cell, change to Tree if needed
        if isinstance(self.nlist, hoomd.md.nlist.Tree):
            self.forcefields[0].nlist = self.nlist(buffer=0.4)
            self.forcefields[0].nlist.exclusions = ["bond", "1-3", "1-4"]
        
        # Set up remaining hoomd objects 
        self._all = hoomd.filter.All()
        gsd_writer, table_file, = self._hoomd_writers(
                group=self._all, sim=self.sim, forcefields=self.forcefields
        )
        self.sim.operations.writers.append(gsd_writer)
        #self.sim.operations.writers.append(table_file)
        self.integrator = hoomd.md.Integrator(dt=self.dt)
        self.integrator.forces = self.forcefields
        self.sim.operations.add(self.integrator)

    def shrink(
            self,
            n_steps,
            kT_init,
            kT_final,
            period,
            tree_nlist=True
    ):
        # Set up temperature ramp during shrinking
        temp_ramp = hoomd.variant.Ramp(
                A=kT_init,
                B=kT_final,
                t_start=self.sim.timestep,
                t_ramp=int(n_steps)
        )
        self.integrator_method = hoomd.md.methods.NVT(
                filter=self._all, kT=temp_ramp, tau=self.tau_kt
        )
        self.sim.operations.integrator.methods = [self.integrator_method]
        #self.integrator.methods = [self.integrator_method]
        #self.sim.operations.add(self.integrator)
        self.sim.state.thermalize_particle_momenta(
                filter=self._all, kT=kT_init
        )

        # Set up box shrinking ramp
        box_resize_trigger = hoomd.trigger.Periodic(period)
        ramp = hoomd.variant.Ramp(
            A=0, B=1, t_start=self.sim.timestep, t_ramp=int(n_steps)
        ) 
        initial_box = self.sim.state.box
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
        self.sim.operations.updaters.append(box_resize)
        
        # Run shrink sim while updating wall potentials 
        if self.wall_axis is not None:
            while self.sim.timestep < n_steps + 1:
                self.sim.run(period)
                self.sim.operations.integrator.forces.remove(self.lj_walls)
                Lx = self.sim.state.box.Lx
                Ly = self.sim.state.box.Ly
                Lz = self.sim.state.box.Lz
                new_walls = self._hoomd_walls(Lx, Ly, Lz)
                self.lj_walls = hoomd.md.external.wall.LJ(walls=new_walls)
                self.lj_walls.params[self.init_snap.particles.types] = {
                        "epsilon": 1.0,
                        "sigma": 1.0,
                        "r_cut": 2.5,
                        "r_extrap": 0
                }
                self.sim.operations.integrator.forces.append(self.lj_walls)
        # Run shrink sim without updating wall potentials
        else:
            self.sim.run(n_steps + 1)
        assert self.sim.state.box == final_box
        self.ran_shrink = True
        
    def quench(
        self,
        n_steps,
        kT=None,
        pressure=None,
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

        """
        if self.wall_axis and pressure is not None:
            raise ValueError(
                    "Wall potentials can only be used with the NVT ensemble."
            )
        if pressure: # Set up NPT Integrator
            self.integrator_method = hoomd.md.methods.NPT(
                    filter=self._all,
                    kT=kT,
                    tau=self.tau_kt,
                    S=pressure,
                    tauS=self.tau_p, 
                    couple="xyz"
            )
            self.sim.operations.integrator.methods = [self.integrator_method]
        else: # Set up (or update) NVT integrator
            if self.ran_shrink:
                self.sim.operations.integrator.methods[0].kT = kT
            else:
                self.integrator_method = hoomd.md.methods.NVT(
                    filter=self._all, kT=kT, tau=self.tau_kt
                )
                self.sim.operations.integrator.methods = [
                        self.integrator_method
                ]
        self.sim.state.thermalize_particle_momenta(filter=self._all, kT=kT)

        try:
            current_timestep = self.sim.timestep
            while self.sim.timestep < n_steps + current_timestep + 1:
                self.sim.run(
                        min(
                            10000,
                            n_steps + current_timestep + 1 - self.sim.timestep
                        )
                )
                if self.wall_time_limit:
                    if (self.sim.device.communicator.walltime + self.sim.walltime >=
                            self.wall_time_limit):
                        break
        finally:
            hoomd.write.GSD.write(
                    state=self.sim.state, mode='wb', filename="restart.gsd"
            ) 
		
    def anneal(
        self,
        kT_init=None,
        kT_final=None,
        pressure=None,
        step_sequence=None,
        schedule=None,
    ):
        """Runs a simulation through a series of temperatures in the 
        NVT or NPT ensemble.
        You can define the annealing sequence by specifying an 
        initial and final temperature and a series of steps, or
        you can pass in a dicitonary of kT:n_steps.

        kT_init : float, optional, default=None
            The starting temperature
        kT_final : float, optional, default=None
            The final temperature
        pressure : float, optional, default=None
            When a pressure is given, the NPT ensemble is used.
        step_sequence : list of ints, optional, default=None
            The series of simulation steps to run between
            kT_init and kT_final
        schedule : dict, optional, default=None
            Use this instead of kT_init, kT_final and step_sequnce to
            explicitly set the series of temperatures and steps at each to run

        """
        if self.wall_axis and pressure is not None:
            raise ValueError(
                "Wall potentials can only be used with the NVT ensemble"
            )

        if not schedule:
            temps = np.linspace(kT_init, kT_final, len(step_sequence))
            temps = [np.round(t, 1) for t in temps]
            schedule = dict(zip(temps, step_sequence))

        if pressure: # Set up NPT Integrator
            self.integrator_method = hoomd.md.methods.NPT(
                    filter=self._all,
                    kT=kT_init,
                    tau=self.tau_kt,
                    S=pressure,
                    tauS=self.tau_p, 
                    couple="xyz"
            )
            self.sim.operations.integrator.methods = [self.integrator_method]
        else: # Add NVT integrator if not already set up
            if not self.ran_shrink:
                self.integrator_method = hoomd.md.methods.NVT(
                    filter=self._all, kT=kT_init, tau=self.tau_kt
                )
                self.sim.operations.integrator.methods = [
                        self.integrator_method
                ]

        for kT in schedule:
            self.sim.operations.integrator.methods[0].kT = kT
            self.sim.state.thermalize_particle_momenta(filter=self._all, kT=kT)
            n_steps = schedule[kT]
            self.sim.run(n_steps) 

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
            The distance along the tensile axis to fix particles in place.
            Treated as a percentage of the initial box length.
            Since particles are fixed on each side, half of fix_ratio
            is used for the distance on each side.

        """
        
        # Set up target volume, tensile axis, etc.
        init_box = self.sim.state.box
        final_box = hoomd.Box(
                Lx=init_box.Lx, Ly=init_box.Ly, Lz=init_box.Lz
        )
        tensile_axis = tensile_axis.lower()
        init_length = getattr(init_box, f"L{tensile_axis}")
        fix_length = init_length * fix_ratio
        target_length = init_length * (1+strain)
        box_resize_trigger = hoomd.trigger.Periodic(expand_period)
        ramp = hoomd.variant.Ramp(
            A=0, B=1, t_start=self.sim.timestep, t_ramp=int(n_steps)
		)
        
        # Set up the walls of fixed particles
        box_max = getattr(init_box, f"L{tensile_axis}")/2
        box_min = -box_max
        if tensile_axis == "x":
            positions = self.init_snap.particles.position[:,0]
            final_box.Lx = target_length
        elif tensile_axis == "y":
            positions = self.init_snap.particles.position[:,1]
            final_box.Ly = target_length
        elif tensile_axis == "z":
            positions = self.init_snap.particles.position[:,2]
            final_box.Lz = target_length
            
        left_tags = np.where(positions < (box_min + fix_length))[0]
        right_tags = np.where(positions > (box_max - fix_length))[0]
        fix_left = hoomd.filter.Tags(left_tags.astype(np.uint32))
        fix_right = hoomd.filter.Tags(right_tags.astype(np.uint32))
        all_fixed = hoomd.filter.Union(fix_left, fix_right)
        integrate_group = hoomd.filter.SetDifference(self._all, all_fixed)

        # Finish setting up simulation
        self.integrator_method = hoomd.md.methods.NVE(filter=integrate_group)
        box_resize = hoomd.update.BoxResize(
                box1=init_box,
                box2=final_box,
                variant=ramp,
                trigger=box_resize_trigger,
                filter=all_fixed
        )
        self.sim.operations.updaters.append(box_resize)
        self.integrator.methods = [self.integrator_method]
        self.sim.state.thermalize_particle_momenta(
                filter=integrate_group, kT=kT
        )

        try:
            while self.sim.timestep < n_steps + 1:
                self.sim.run(min(10000, n_steps + 1 - self.sim.timestep))
                if self.wall_time_limit:
                    if (self.sim.device.communicator.walltime +
                            self.sim.walltime >=
                            self.wall_time_limit):
                        break
        finally:
            hoomd.write.GSD.write(
                    state=self.sim.state, mode='wb', filename="restart.gsd"
            )
        
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
        return gsd_writer, None
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

        Similar to the `create_hoomd_forcefield` function
        from mBuild, but designed to work when initializing
        a system from a gsd file rather than a Parmed structure.
        Created specifically for using table potentials with
        coarse-grained systems.

        """
        if self.restart is None:
            with gsd.hoomd.open(self.system, "rb") as f:
                init_snap = f[0]
        else:
            with gsd.hoomd.open(self.restart) as f:
                init_snap = f[-1]
                print("Simulation initialized from restart file")
        # Create pair table potentials
        nlist = self.nlist(buffer=0.4)
        pair_table = hoomd.md.pair.Table(nlist=nlist)
        for pair in [list(i) for i in combo(init_snap.particles.types, r=2)]:
            _pair = "-".join(sorted(pair))
            pair_pot_file = f"{self.cg_ff_path}/{_pair}.txt"
            try:
                assert os.path.exists(pair_pot_file)
            except AssertionError:
                raise RuntimeError(f"The potential file {pair_pot_file} "
                    f"for pair {_pair} does not exist in {self.cg_ff_path}."
                )
            pair_data = np.loadtxt(pair_pot_file)
            r_min = pair_data[:,0][0]
            r_cut = pair_data[:,0][-1]
            pair_table.params[tuple(sorted(pair))] = dict(
                    r_min=r_min, U=pair_data[:,1], F=pair_data[:,2]
            )
            pair_table.r_cut[tuple(sorted(pair))] = r_cut

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

        bond_table = hoomd.md.bond.Table(width=bond_pot_widths[0])
        for bond, fpath in zip(bonds, bond_pot_files):
            bond_data = np.loadtxt(fpath)
            r_min = bond_data[:,0][0]
            r_max = bond_data[:,0][-1]
            bond_table.params[bond] = dict(
                    r_min=r_min,
                    r_max=r_max,
                    U=bond_data[:,1],
                    F=bond_data[:,2]
            )
        
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

        angle_table = hoomd.md.angle.Table(width=angle_pot_widths[0])
        for angle, fpath in zip(angles, angle_pot_files):
            angle_data = np.loadtxt(fpath)
            angle_table.params[angle] = dict(
                    U=angle_data[:,1], tau=angle_data[:,2]
            )

        hoomd_forces = [
                pair_table,
                bond_table,
                angle_table,
        ]
        return init_snap, hoomd_forces 

    def _hoomd_walls(self, Lx, Ly, Lz):
        """Create hoomd LJ wall potentials"""
        wall_origin = np.asarray(self.wall_axis) * np.array(
                [Lx/2, Ly/2, Lz/2]
        )
        normal_vector = -np.asarray(self.wall_axis)
        wall_origin2 = -wall_origin
        normal_vector2 = -normal_vector
        wall1 = hoomd.wall.Plane(origin=wall_origin, normal=normal_vector)
        wall2 = hoomd.wall.Plane(origin=wall_origin2, normal=normal_vector2)
        return [wall1, wall2] 
