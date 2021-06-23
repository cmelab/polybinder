import json
import operator
import os
import random
import time
from collections import namedtuple

import ele
import foyer
import gsd
import hoomd
import hoomd.md
import mbuild as mb
import numpy as np
import scipy.optimize
from foyer import Forcefield
from hoomd.md import wall
from mbuild.formats.hoomd_simulation import create_hoomd_simulation
from mbuild.lib.recipes import Polymer
from scipy.special import gamma

from uli_init.compounds import COMPOUND_DIR
from uli_init.forcefields import FF_DIR
from uli_init.utils import base_units

units = base_units.base_units()


class Simulation:
    def __init__(
        self,
        system,
        target_box=None,
        r_cut=1.2,
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
    ):

        self.system_pmd = system.system  # Parmed structure
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

        if ref_units and not auto_scale:
            self.ref_energy = ref_units["energy"]
            self.ref_distance = ref_units["distance"]
            self.ref_mass = ref_units["mass"]

        # Pulled from mBuild hoomd_simulation.py
        elif auto_scale and not ref_units:
            self.ref_mass = max([atom.mass for atom in self.system_pmd.atoms])
            pair_coeffs = list(
                set(
                    (atom.type, atom.epsilon, atom.sigma)
                    for atom in self.system_pmd.atoms
                )
            )
            self.ref_energy = max(pair_coeffs, key=operator.itemgetter(1))[1]
            self.ref_distance = max(pair_coeffs, key=operator.itemgetter(2))[2]

        if system.type != "interface":
            # nm
            self.reduced_target_L = system.target_L / self.ref_distance
            # angstroms
            self.reduced_init_L = (self.system_pmd.box[0] / self.ref_distance)

            if target_box:
                self.target_box = target_box
            else:
                self.target_box = [self.reduced_target_L] * 3

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

    def quench(
        self,
        n_steps,
        kT=None,
        pressure=None,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
        walls=True,
    ):
        """"""
        if walls and pressure:
            raise ValueError(
                    "Wall potentials can only be used with the NVT ensemble"
                    )
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
            if len([i for i in (shrink_kT, shrink_steps) if i is None]) == 1:
                raise ValueError(
                "Both of  shrink_kT and shrink_steps need to be given"
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
                    (shrink_steps, self.target_box[0] * 10)
                ])
                y_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Ly),
                    (shrink_steps, self.target_box[1] * 10)
                ])
                z_variant = hoomd.variant.linear_interp([
                    (0, init_snap.box.Lz),
                    (shrink_steps, self.target_box[2] * 10)
                ])
                box_updater = hoomd.update.box_resize(
                    Lx=x_variant,
                    Ly=y_variant,
                    Lz=z_variant,
                    period=shrink_period
                )

                # Update wall origins during shrinking
                if walls:
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
                    step = 0
                    start = time.time()
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
                        print(f"Finished step {step} of {shrink_steps}")
                        print(f"Shrinking is {round(step / shrink_steps, 5) * 100}% complete")
                        print(f"time elapsed: {time.time() - start}")
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
                try:
                    integrator
                except NameError:
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
        walls=True,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
    ):
        if walls and pressure:
            raise ValueError(
                    "Wall potentials can only be used with the NVT ensemble"
                    )
        if not schedule:
            temps = np.linspace(kT_init, kT_final, len(step_sequence))
            temps = [np.round(t, 1) for t in temps]
            schedule = dict(zip(temps, step_sequence))

        # Get hoomd stuff set:
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

            if shrink_kT and shrink_steps:
                integrator = hoomd.md.integrate.nvt(
                        group=_all,
                        tau=self.tau_kt,
                        kT=shrink_kT
                        )
                integrator.randomize_velocities(seed=self.seed)

                x_variant = hoomd.variant.linear_interp([
                    (0, self.reduced_init_L),
                    (shrink_steps, self.target_box[0] * 10)
                ])
                y_variant = hoomd.variant.linear_interp([
                    (0, self.reduced_init_L),
                    (shrink_steps, self.target_box[1] * 10)
                ])
                z_variant = hoomd.variant.linear_interp([
                    (0, self.reduced_init_L),
                    (shrink_steps, self.target_box[2] * 10)
                ])
                box_updater = hoomd.update.box_resize(
                    Lx=x_variant,
                    Ly=y_variant,
                    Lz=z_variant,
                    period=shrink_period
                )
                if walls:
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

