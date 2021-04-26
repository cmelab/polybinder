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
        tau=0.1,
        dt=0.0001,
        auto_scale=True,
        ref_units=None,
        nlist="cell",
        mode="gpu",
        gsd_write=1e4,
        log_write=1e3,
        seed=42,
    ):

        self.system = system
        self.system_pmd = system.system_pmd  # Parmed structure
        self.r_cut = r_cut
        self.e_factor = e_factor
        self.tau = tau
        self.dt = dt
        self.auto_scale = auto_scale
        self.ref_units = ref_units
        self.nlist = nlist
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

        if system.type == "melt":
            # nm
            self.reduced_target_L = self.system.target_L / self.ref_distance
            # angstroms
            self.reduced_init_L = (self.system_pmd.box[0] / self.ref_distance)

            # TODO: Use target_box to generate non-cubic simulation volumes
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
        kT,
        n_steps,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
        walls=True,
    ):
        """"""
        hoomd_args = f"--single-mpi --mode={self.mode}"
        sim = hoomd.context.initialize(hoomd_args)
        with sim:
            objs, refs = create_hoomd_simulation(
                structure=self.system_pmd,
                ref_distance=self.ref_distance,
                ref_mass=self.ref_mass,
                ref_energy=self.ref_energy,
                r_cut=self.r_cut,
                n_list=self.nlist,
                auto_scale=self.auto_scale,
            )
            hoomd_system = objs[1]
            init_snap = objs[0]
            _all = hoomd.group.all()
            hoomd.md.integrate.mode_standard(dt=self.dt)
            integrator = hoomd.md.integrate.nvt(group=_all, kT=kT, tau=self.tau)
            integrator.randomize_velocities(seed=self.seed)

            # LJ walls set on each side along x-axis
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

            if shrink_kT and shrink_steps:
                shrink_gsd = hoomd.dump.gsd(
                    "traj-shrink.gsd",
                    period=self.gsd_write,
                    group=_all,
                    phase=0,
                    overwrite=True,
                )

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

                integrator.set_params(kT=shrink_kT)  # shrink temp
                integrator.randomize_velocities(seed=self.seed)

                # Update wall origins during shrinking
                if walls:
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
                shrink_gsd.disable()
                box_updater.disable()
            # Set up new gsd and log dumps for actual simulation
            hoomd.dump.gsd(
                "sim_traj.gsd",
                period=self.gsd_write,
                group=_all,
                phase=0,
                overwrite=True,
            )
            gsd_restart = hoomd.dump.gsd(
                "restart.gsd",
                period=self.gsd_write,
                group=_all,
                truncate=True,
                phase=0
            )
            hoomd.analyze.log(
                "sim_traj.log",
                period=self.log_write,
                quantities=self.log_quantities,
                header_prefix="#",
                overwrite=True,
                phase=0,
            )
            # Run the primary simulation
            integrator.set_params(kT=kT)
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
        step_sequence=None,
        schedule=None,
        walls=True,
        shrink_kT=None,
        shrink_steps=None,
        shrink_period=None,
    ):

        if not schedule:
            temps = np.linspace(kT_init, kT_final, len(step_sequence))
            temps = [np.round(t, 1) for t in temps]
            schedule = dict(zip(temps, step_sequence))

        # Get hoomd stuff set:
        hoomd_args = f"--single-mpi --mode={self.mode}"
        sim = hoomd.context.initialize(hoomd_args)
        with sim:
            objs, refs = create_hoomd_simulation(
                structure=self.system_pmd,
                ref_distance=self.ref_distance,
                ref_mass=self.ref_mass,
                ref_energy=self.ref_energy,
                r_cut=self.r_cut,
                n_list=self.nlist,
                auto_scale=self.auto_scale,
            )
            hoomd_system = objs[1]
            init_snap = objs[0]
            _all = hoomd.group.all()
            hoomd.md.integrate.mode_standard(dt=self.dt)
            integrator = hoomd.md.integrate.nvt(
                    group=_all,
                    kT=kT_init,
                    tau=self.tau
                    )
            integrator.randomize_velocities(seed=self.seed)

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

            if shrink_kT and shrink_steps:
                shrink_gsd = hoomd.dump.gsd(
                    "traj-shrink.gsd",
                    period=self.gsd_write,
                    group=_all,
                    phase=0,
                    overwrite=True,
                )

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

                integrator.set_params(kT=shrink_kT)  # shrink temp
                integrator.randomize_velocities(seed=self.seed)

                if walls:
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
                shrink_gsd.disable()
                box_updater.disable()
            # Set up new log and gsd files for simulation:
            hoomd.dump.gsd(
                "sim_traj.gsd",
                period=self.gsd_write,
                group=_all,
                phase=0,
                overwrite=True,
            )
            gsd_restart = hoomd.dump.gsd(
                "restart.gsd",
                period=self.gsd_write,
                group=_all,
                truncate=True,
                phase=0
            )
            hoomd.analyze.log(
                "sim_traj.log",
                period=self.log_write,
                quantities=self.log_quantities,
                header_prefix="#",
                overwrite=True,
                phase=0,
            )

            for kT in schedule:  # Start iterating through annealing steps
                print(f"Running @ Temp = {kT} kT")
                n_steps = schedule[kT]
                print(f"Running for {n_steps} steps")
                integrator.set_params(kT=kT)
                integrator.randomize_velocities(seed=self.seed)
                try:
                    hoomd.run(n_steps)
                except hoomd.WalltimeLimitReached:
                    pass
                finally:
                    gsd_restart.write_restart()


class Interface:
    def __init__(
        self,
        slabs,
        ref_distance=None,
        gap=0.1,
        forcefield="gaff",
    ):
        self.forcefield = forcefield
        self.type = "interface"
        self.ref_distance = ref_distance
        if not isinstance(slabs, list):
            slabs = [slabs]
        if len(slabs) == 2:
            slab_files = slabs
        else:
            slab_files = slabs * 2

        interface = mb.Compound()
        slab_1 = self._gsd_to_mbuild(slab_files[0], self.ref_distance)
        slab_2 = self._gsd_to_mbuild(slab_files[1], self.ref_distance)
        interface.add(new_child=slab_1, label="left")
        interface.add(new_child=slab_2, label="right")
        x_len = interface.boundingbox.lengths[0]
        interface["left"].translate((-x_len - gap, 0, 0))

        system_box = mb.box.Box(
                mins=(0, 0, 0),
                maxs=interface.boundingbox.lengths
                )
        system_box.maxs[0] += 2 * self.ref_distance * 1.1225
        interface.box = system_box
        # Center in the adjusted box
        interface.translate_to([
            interface.box.maxs[0] / 2,
            interface.box.maxs[1] / 2,
            interface.box.maxs[2] / 2,
        ])

        if forcefield == "gaff":
            ff_path = f"{FF_DIR}/gaff-nosmarts.xml"
            forcefield = foyer.Forcefield(forcefield_files=ff_path)
        self.system_pmd = forcefield.apply(interface)

    def _gsd_to_mbuild(self, gsd_file, ref_distance):
        element_mapping = {
            "oh": "O",
            "ca": "C",
            "os": "O",
            "o": "O",
            "c": "C",
            "ho": "H",
            "ha": "H",
        }
        snap = trajectory = gsd.hoomd.open(gsd_file)[-1]
        pos_wrap = snap.particles.position * ref_distance
        atom_types = [snap.particles.types[i] for i in snap.particles.typeid]
        elements = [element_mapping[i] for i in atom_types]

        comp = mb.Compound()
        for pos, element, atom_type in zip(pos_wrap, elements, atom_types):
            child = mb.Compound(name=f"_{atom_type}", pos=pos, element=element)
            comp.add(child)

        bonds = [(i, j) for (i, j) in snap.bonds.group]
        self._add_bonds(compound=comp, bonds=bonds)
        return comp

    def _add_bonds(self, compound, bonds):
        particle_dict = {}
        for idx, particle in enumerate(compound.particles()):
            particle_dict[idx] = particle

        for (i, j) in bonds:
            atom1 = particle_dict[i]
            atom2 = particle_dict[j]
            compound.add_bond(particle_pair=[atom1, atom2])


class System:
    def __init__(
        self,
        molecule,
        para_weight,
        density,
        n_compounds=None,
        polymer_lengths=None,
        forcefield=None,
        epsilon=1e-7,
        sample_pdi=False,
        pdi=None,
        Mn=None,
        Mw=None,
        mass_dist_type="weibull",
        remove_hydrogens=False,
        assert_dihedrals=True,
        seed=24,
        expand_factor=5
    ):
        self.molecule = molecule
        self.para_weight = para_weight
        self.density = density
        self.target_L = None
        self.remove_hydrogens = remove_hydrogens
        self.epsilon = epsilon
        self.forcefield = forcefield
        self.assert_dihedrals = assert_dihedrals
        self.seed = seed
        self.system_mass = 0
        self.para = 0
        self.meta = 0
        self.type = "melt"
        self.expand_factor = expand_factor

        if sample_pdi:
            if isinstance(n_compounds, int):
                self.n_compounds = n_compounds
            elif isinstance(n_compounds, list) and len(n_compounds) == 1:
                self.n_compounds = n_compounds[0]
            elif isinstance(n_compounds, list) and len(n_compounds) != 1:
                raise TypeError(
                    "n_compounds should be of length 1 when sample_pdi is True."
                )
            pdi_arg_sum = sum([x is not None for x in [pdi, Mn, Mw]])
            assert (
                pdi_arg_sum >= 2
            ), "At least two of [pdi, Mn, Mw] must be given."
            if pdi_arg_sum == 3:
                # special case, make sure that pdi = Mw / Mn
                assert (
                    abs(pdi - (Mw / Mn)) < self.epsilon
                ), "PDI value does not match Mn and Mw values."
            else:
                # need to recover one of Mw or Mn or pdi
                if Mn is None:
                    Mn = Mw / pdi
                if Mw is None:
                    Mw = pdi * Mn
                if pdi is None:
                    pdi = Mw / Mn
            self.Mn = Mn
            self.Mw = Mw
            self.pdi = pdi

            # this returns a numpy.random callable set up with
            # recovered parameters
            mass_distribution_dict = self._recover_mass_dist(mass_dist_type)
            self.mass_sampler = mass_distribution_dict["sampler"]
            self.mass_distribution = mass_distribution_dict["functional_form"]
            # TODO: make sure we don't sample any negative weights
            samples = np.round(self.mass_sampler(n_compounds)).astype(int)
            # get all unique lengths in increasing order
            self.polymer_lengths = sorted(list(set(samples)))
            # get count of each length
            self.n_compounds = [
                    list(samples).count(x) for x in self.polymer_lengths
                    ]
            print(
                f"polymer_lengths: {self.polymer_lengths},",
                " n_compounds: {self.n_compounds}"
            )

        else:  # Do some validation, get things in the correct data types
            if not isinstance(n_compounds, list):
                self.n_compounds = [n_compounds]
            else:
                self.n_compounds = n_compounds

            if not isinstance(polymer_lengths, list):
                self.polymer_lengths = [polymer_lengths]
            else:
                self.polymer_lengths = polymer_lengths

        if len(self.n_compounds) != len(self.polymer_lengths):
            raise ValueError(
                    "n_compounds and polymer_lengths should be equal length"
                    )

        # mBuild object before applying FF
        self.system_mb = self._pack()
        if self.forcefield:
            # parmed object after applying FF
            self.system_pmd = self._type_system()

    def _weibull_k_expression(self, x):
        return (
                (2.0 * x * gamma(2.0 / x)) /
                gamma(1.0 / x) ** 2 - (self.Mw / self.Mn)
                )

    def _weibull_lambda_expression(self, k):
        return self.Mn * k / gamma(1.0 / k)

    def _recover_mass_dist(self, distribution="Gaussian"):
        """This function takes in two of the three quantities [Mn, Mw, PDI],
        and fits either a Gaussian or Weibull distribution of molar masses to
        them.
        """
        distribution = distribution.lower()

        if distribution != "gaussian" and distribution != "weibull":
            raise ValueError(
                'Molar mass distribution must be "gaussian" or "weibull".'
            )
        if distribution == "gaussian":
            mean = self.Mn
            sigma = self.Mn * (self.Mw - self.Mn)
            return {
                "sampler": lambda N: np.random.normal(
                    loc=mean, scale=sigma, size=N
                    ),
                "functional_form": lambda x: np.exp(
                    -((x - Mn) ** 2) / (2.0 * sigma)
                    ),
            }
        # Weibull
        # get the shape parameter
        a = scipy.optimize.root(self._weibull_k_expression, x0=1.0)
        recovered_k = a["x"]
        # get the scale parameter
        recovered_lambda = self._weibull_lambda_expression(recovered_k)
        return {
            "sampler": lambda N: recovered_lambda
            * np.random.weibull(recovered_k, size=N),
            "functional_form": lambda x: recovered_k
            / recovered_lambda
            * (x / recovered_lambda) ** (recovered_k - 1)
            * np.exp(-((x / recovered_lambda) ** recovered_k)),
        }

    def _pack(self):
        random.seed(self.seed)
        mb_compounds = []
        for _length, _n in zip(self.polymer_lengths, self.n_compounds):
            for i in range(_n):
                polymer, sequence = build_molecule(
                    self.molecule, _length, self.para_weight
                )

                mb_compounds.append(polymer)
                self.para += sequence.count("para")
                self.meta += sequence.count("meta")
            mass = _n * np.sum(
                ele.element_from_symbol(p.name).mass
                for p in polymer.particles()
            )
            self.system_mass += mass  # amu

        # Figure out correct box dimensions and expand the box to make the
        # PACKMOL step faster. Will shrink down to accurate L during simulation
        if len(mb_compounds) == 1:
            L = self._calculate_L()
            system = mb_compounds[0]
            system.Box = mb.box.Box(mins=system.boundingbox.mins,
                    maxs=system.boundingbox.maxs)
            expand_factor = np.array([L,L,L]) / system.boundingbox.lengths
            system.Box.mins *= expand_factor
            system.Box.maxs *= expand_factor
        else:
            L = self._calculate_L() * self.expand_factor
            system = mb.packing.fill_box(
                compound=mb_compounds,
                n_compounds=[1 for i in mb_compounds],
                box=[L, L, L],
                overlap=0.2,
                edge=0.9,
                fix_orientation=True,
            )
            system.Box = mb.box.Box([L, L, L])
        return system

    def _type_system(self):
        if self.forcefield == "gaff":
            ff_path = f"{FF_DIR}/gaff.xml"
            forcefield = foyer.Forcefield(forcefield_files=ff_path)
        elif self.forcefield == "opls":
            forcefield = foyer.Forcefield(name="oplsaa")

        typed_system = forcefield.apply(
            self.system_mb, assert_dihedral_params=self.assert_dihedrals
        )
        if self.remove_hydrogens:
            typed_system.strip(
                    [a.atomic_number == 1 for a in typed_system.atoms]
                    )
        return typed_system

    def _calculate_L(self):
        """
        Calcualte the box length needed for entered density
        Right now, assuming cubic box
        Return L in nm (mBuild units)
        """
        M = self.system_mass * units["amu_to_g"]  # grams
        L = (M / self.density) ** (1 / 3)  # centimeters
        L *= units["cm_to_nm"]  # convert cm to nm
        self.target_L = L  # Used during shrink step
        return L


def build_molecule(molecule, length, para_weight):
    """
    `build_molecule` uses SMILES strings to build up a polymer from monomers.
    The configuration of each monomer is determined by para_weight and the
    random_sequence() function.
    
    Parameters
    ----------
    molecule : str
        The monomer molecule to be used to build up the polymer.
        Available options are limited  to the .json files in the compounds
        directory
        Use the molecule name as seen in the .json file without including .json
    length : int
        The number of monomer units in the final polymer molecule
    para_weight : float, limited to values between 0 and 1
        The relative amount of para configurations compared to meta.
        Passed into random_sequence() to determine the monomer sequence of the
        polymer.
        A 70/30 para to meta system would require a para_weight = 0.70

    Returns
    -------
    compound : mBuild.Compound
        An instance of the single polymer created
    sequence : list
        List of the configuration sequence of the finished compound
    """
    f = open(f"{COMPOUND_DIR}/{molecule}.json")
    mol_dict = json.load(f)
    f.close()

    monomer_sequence = random_sequence(para_weight, length)
    compound = Polymer()
    para = mb.load(mol_dict["para_smiles"], smiles=True, backend="rdkit")
    meta = mb.load(mol_dict["meta_smiles"], smiles=True, backend="rdkit")

    if len(set(monomer_sequence)) == 2: # Copolymer
        compound.add_monomer(meta, 
                mol_dict["meta_bond_indices"],
                mol_dict["bond_distance"],
                replace=True
            )
        compound.add_monomer(para,
                mol_dict["para_bond_indices"],
                mol_dict["bond_distance"],
                replace=True
            )
    else:
        if monomer_sequence[0] == "P": # Only para
            compound.add_monomer(para,
                    mol_dict["para_bond_indices"],
                    mol_dict["bond_distance"],
                    replace=True
                )
        elif monomer_sequence[0] == "M": # Only meta
            compound.add_monomer(meta,
                    mol_dict["meta_bond_indices"],
                    mol_dict["bond_distance"],
                    replace=True
                )

    compound.build(n=1, sequence=monomer_sequence, add_hydrogens=True)
    return compound, monomer_sequence


def random_sequence(para_weight, length):
    """
    random_sequence returns a list containing a random sequence of strings
    'P' and 'M'.
    This is used by build_molecule() to create a polymers chains.

    Parameters:
    -----------
    para_weight : float, limited to values between 0 and 1
        The relative amount of para configurations compared to meta.
        Defined in build_molecule()
    length : int
        The number of elements in the random sequence.
        Defined in build_molecule()
    """
    meta_weight = 1 - para_weight
    options = ["P", "M"]
    probability = [para_weight, meta_weight]
    sequence = random.choices(options, weights=probability, k=length)
    return sequence
