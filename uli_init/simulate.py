import operator
from itertools import combinations_with_replacement as combo
import gsd
import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np
import parmed as pmd
from hoomd.md import wall
from mbuild.formats.hoomd_simulation import create_hoomd_simulation
from uli_init.library import COMPOUND_DIR, SYSTEM_DIR, FF_DIR


class Simulation:
    def __init__(
        self,
        system,
        r_cut=2.5,
        e_factor=0.5,
        tau_kt=0.1,
        tau_p=None,
        nlist="cell",
        dt=0.0001,
        auto_scale=True,
        ref_units=None,
        mode="gpu",
        gsd_write=1e4,
        log_write=1e3,
        seed=42,
        bond_dict=None,
        angle_dict=None
    ):

        self.r_cut = r_cut
        self.e_factor = e_factor
        self.tau_kt = tau_kt
        self.tau_p = tau_p
        self.nlist = getattr(hoomd.md.nlist, nlist.lower())
        self.dt = dt
        self.auto_scale = auto_scale
        self.ref_units = ref_units
        self.mode = mode
        self.gsd_write = gsd_write
        self.log_write = log_write
        self.seed = seed
        self.bond_dict = bond_dict
        self.angle_dict = angle_dict

        if isinstance(system.system, gsd.hoomd.snapshot):
            assert ref_units != None, (
                    "Autoscaling is not supported for coarse-grained systems."
                    "Provide the relevant reference units"
                    )
            assert all([self.bond_dict, self.angle_dict]), (
                    "If using a coarse-grain system, pass in the bonding "
                    "and angle information via the bond_dict and angle_dict "
                    "parameters."
                    )
            self.cg_system = True
            self.ref_energy = ref_units["energy"]
            self.ref_distance = ref_units["distance"]
            self.ref_mass = ref_units["mass"]
            self.system = system.system
        elif isinstance(system.system, pmd.Structure):
            self.system = system.system
            self.cg_system = False
            if ref_units and not auto_scale:
                self.ref_energy = ref_units["energy"]
                self.ref_distance = ref_units["distance"]
                self.ref_mass = ref_units["mass"]
            # Pulled from mBuild hoomd_simulation.py
            elif auto_scale and not ref_units:
                self.ref_mass = max(
                        [atom.mass for atom in self.system_pmd.atoms]
                        )
                pair_coeffs = list(
                set(
                    (atom.type, atom.epsilon, atom.sigma)
                    for atom in self.system_pmd.atoms
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
            "volume",
            "potential_energy",
            "kinetic_energy",
            "pair_lj_energy",
            "bond_harmonic_energy",
            "angle_harmonic_energy",
        ]


    def create_hoomd_sim_from_snapshot(self):
        hoomd_system = hoomd.init.read_snapshot(self.system)
        table = hoomd.md.pair.table(width=101, nlist=self.nlist)
        for pair in [list(i) for i in combo(self.system.particles.types, r=2)]:
            pair.sort()
            _pair = "-".join(pair)
            table_pot_file = f"{FF_DIR}/{_pair}.txt"
            table.set_from_file(
                f"{pair[0]}", "{pair[1]}", filename='{table_pot_file}'
            )
        # Create bond and angle objects 
        harmonic_bond = hoomd.md.bond.harmonic()
        harmonic_angle = hoomd.md.angle.harmonic()
        for bond in self.bond_dicts:
            bond_pair = [bond["type1"], bond["type2"].sort()
            name = "-".join(bond_pair)
            k, r0 = bond["k"], bond["r0"]
            harmonic_bond.bond_coeff.set(name, k, r0)
        for angle in self.angle_dicts:
            name = "-".join(
                    angle["type1"], angle["type2"], angle["type3"]
                )
            k, theta0 = angle["k"], angle["theta0"]
            harmonic_angle.angle_coeff.set(
                    name, k, theta0
                )
        hoomd_objs = [
                self.system,
                hoomd_system,
                self.nlist,
                table,
                harmonic_bond,
                harmonic_angle,
            ]
                
        return hoomd_objs 

        

    def quench(
        self,
        n_steps,
        kT=None,
        pressure=None,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
        use_walls=True,
    ):
        """"""
        if use_walls and pressure:
            raise ValueError(
                    "Wall potentials can only be used with the NVT ensemble."
                    )
        if [shrink_kT, shrink_steps, shrink_period].count(None) %3 != 0:
            raise ValueError(
            "If shrinking, all of  shrink_kT, shrink_steps and "
            "shrink_periopd need to be given."
        )

        hoomd_args = f"--single-mpi --mode={self.mode}"
        sim = hoomd.context.initialize(hoomd_args)
        with sim:
            if self.cg_system is False:
                objs, refs = create_hoomd_simulation(
                    self.system_pmd,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
                    nlist=self.nlist
                )
            elif self.cg_system is True:
                objs = create_hoomd_sim_from_snapshot()

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

            if use_walls:
                wall_origin = (init_snap.box.Lx / 2, 0, 0)
                normal_vector = (-1, 0, 0)
                wall_origin2 = (-init_snap.box.Lx / 2, 0, 0)
                normal_vector2 = (1, 0, 0)
                walls = wall.group(
                    wall.plane(
                        origin=wall_origin, normal=normal_vector, inside=True
                        ),
                    wall.plane(
                        origin=wall_origin2, normal=normal_vector2, inside=True
                        ),
                )
                wall_force = wall.lj(walls, r_cut=2.5)
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

                x_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Lx),
                    (shrink_steps, self.target_box[0])
                ])
                y_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Ly),
                    (shrink_steps, self.target_box[1])
                ])
                z_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Lz),
                    (shrink_steps, self.target_box[2])
                ])
                box_updater = hoomd.update.box_resize(
                    Lx=x_variant,
                    Ly=y_variant,
                    Lz=z_variant,
                    period=shrink_period
                )

                # Update wall origins during shrinking
                momentum = hoomd.md.update.zero_momentum(period=shrink_steps)
                if use_walls:
                    step = 0
                    while step < shrink_steps:
                        hoomd.run_upto(step + shrink_period)
                        current_box = hoomd_system.box
                        walls.del_plane([0, 1])
                        walls.add_plane(
                                (current_box.Lx / 2, 0, 0), normal_vector
                                )
                        walls.add_plane(
                                (-current_box.Lx / 2, 0, 0),
                                normal_vector2
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
                hoomd.run(n_steps)
            except hoomd.WalltimeLimitReached:
                pass
            finally:
                gsd_restart.write_restart()

    def anneal(
        self,
        kT_init=None,
        kT_final=None,
        pressure=None,
        step_sequence=None,
        schedule=None,
        use_walls=True,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
    ):
        if use_walls and pressure:
            raise ValueError(
                    "Wall potentials can only be used with the NVT ensemble"
                    )
        if [shrink_kT, shrink_steps, shrink_period].count(None) %3 != 0:
            raise ValueError(
                    "If shrinking, then all of shirnk_kT, shrink_steps "
                    "and shrink_period need to be given"
                    )
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
                    self.system_pmd,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
                    nlist=self.nlist
                )
            elif self.cg_system is True:
                objs = create_hoomd_sim_from_snapshot()

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
            if use_walls:
                wall_origin = (init_snap.box.Lx / 2, 0, 0)
                normal_vector = (-1, 0, 0)
                wall_origin2 = (-init_snap.box.Lx / 2, 0, 0)
                normal_vector2 = (1, 0, 0)
                walls = wall.group(
                    wall.plane(
                        origin=wall_origin, normal=normal_vector, inside=True
                        ),
                    wall.plane(
                        origin=wall_origin2, normal=normal_vector2, inside=True
                        )
                )

                wall_force = wall.lj(walls, r_cut=2.5)
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

                x_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Lx),
                    (shrink_steps, self.target_box[0])
                ])
                y_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Ly),
                    (shrink_steps, self.target_box[1])
                ])
                z_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Lz),
                    (shrink_steps, self.target_box[2])
                ])
                box_updater = hoomd.update.box_resize(
                    Lx=x_variant,
                    Ly=y_variant,
                    Lz=z_variant,
                    period=shrink_period
                )
                # Update walls due to shrink box changes
                if use_walls:
                    step = 0
                    while step < shrink_steps:
                        hoomd.run_upto(step + shrink_period)
                        current_box = hoomd_system.box
                        walls.del_plane([0, 1])
                        walls.add_plane(
                                (current_box.Lx / 2, 0, 0), normal_vector
                                )
                        walls.add_plane(
                                (-current_box.Lx / 2, 0, 0), normal_vector2
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

            for kT in schedule: 
                n_steps = schedule[kT]
                integrator.set_params(kT=kT)
                integrator.randomize_velocities(seed=self.seed)
                print(f"Running @ Temp = {kT} kT")
                print(f"Running for {n_steps} steps")
                try:
                    hoomd.run(n_steps)
                except hoomd.WalltimeLimitReached:
                    pass
                finally:
                    gsd_restart.write_restart()

    def tensile(self,
            kT,
            strain,
            n_steps,
            expand_period,
            scale=False,
            x_fix=0.05
            ):
        """
        """
        hoomd_args = f"--single-mpi --mode={self.mode}"
        sim = hoomd.context.initialize(hoomd_args)
        with sim:
            objs, refs = create_hoomd_simulation(
                    self.system_pmd,
                    self.ref_distance,
                    self.ref_mass,
                    self.ref_energy,
                    self.r_cut,
                    self.auto_scale,
                    nlist=self.nlist
                )
            hoomd_system = objs[1]
            init_snap = objs[0]
            # Set up groups, based in fix length along X
            fix_length = (init_snap.box.Lx / 2) * x_fix
            fix_left = hoomd.group.cuboid( # Negative x side of box
                    name="left",
                    xmin=-init_snap.box.Lx / 2,
                    xmax=(-init_snap.box.Lx / 2)  + fix_length
                )
            # Positive x side of box
            fix_right = hoomd.group.cuboid(
                    name="right",
                    xmin=(init_snap.box.Lx / 2) - fix_length,
                    xmax=init_snap.box.Lx / 2
                )
            _all_fixed = hoomd.group.union(
                    name="fixed", a=fix_left, b=fix_right
                )
            _all = hoomd.group.all()
            _integrate = hoomd.group.difference(
                    name="integrate", a=_all, b=_all_fixed
                    )
            assert(
                len([p for p in _integrate]) == (len([p for p in _all]) -
                    len([p for p in _all_fixed])
                    )
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
            hoomd.dump.gsd(
                    "fixed_traj.gsd",
                    period=self.gsd_write,
                    group=_all_fixed,
                    phase=0,
                    dynamic=["momentum"],
                    overwrite=False
                )
            hoomd.dump.gsd(
                    "not_fixed_traj.gsd",
                    period=self.gsd_write,
                    group=_integrate,
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
            
            # Set up volume expansion
            target_length = init_snap.box.Lx*(1 + strain)
            x_variant = hoomd.variant.linear_interp(
                [(0, init_snap.box.Lx), (n_steps, target_length)]
            )
            box_updater = hoomd.update.box_resize(
                    Lx=x_variant, period=expand_period, scale_particles=scale
                    )
            # Start simulation run
            step = 0
            last_Lx = init_snap.box.Lx
            while step < n_steps:
                try:
                    hoomd.run_upto(step + expand_period)
                    current_box = hoomd_system.box
                    diff = current_box.Lx - last_Lx
                    for particle in fix_left:
                        particle.position = (
                                    particle.position[0] - (diff / 2),
                                    particle.position[1],
                                    particle.position[2]
                                )
                    for particle in fix_right:
                        particle.position = (
                                    particle.position[0] + (diff / 2),
                                    particle.position[1],
                                    particle.position[2]
                                )

                    step += expand_period
                    last_Lx = current_box.Lx
                except hoomd.WalltimeLimitReached:
                    pass
                finally:
                    gsd_restart.write_restart()
