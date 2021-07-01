from collections import namedtuple
import json
import operator
import os
import random
import time
from warnings import warn

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
from mbuild.coordinate_transform import z_axis_transform
from scipy.special import gamma

from uli_init.library import COMPOUND_DIR, SYSTEM_DIR, FF_DIR
from uli_init.utils import base_units

units = base_units.base_units()


class System:
    def __init__(
        self,
        system_type,
        density,
        molecule=None,
        n_compounds=None,
        polymer_lengths=None,
        para_weight=None,
        monomer_sequence=None,
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
        expand_factor=5,
        **kwargs
    ):
        self.type = system_type
        self.molecule = molecule
        self.para_weight = para_weight
        self.monomer_sequence = monomer_sequence
        self.density = density
        self.forcefield = forcefield
        self.remove_hydrogens = remove_hydrogens
        self.assert_dihedrals = assert_dihedrals
        self.seed = seed
        self.expand_factor = expand_factor
        self.target_L = None
        self.system_mass = 0
        self.para = 0
        self.meta = 0
        self.molecule_sequences = []
        
        if self.type == "custom":
            if any((molecule,
                para_weight,
                monomer_sequence,
                n_compounds,
                polymer_lengths)):
                warn("If a custom created system is being used, then "
                     "the system generation parameters of `molecule, "
                     "para_weight, n_compounds, polymer_lengths, and "
                     "monomer_sequence will be ignored."
                        )
                self.molecule = None
                self.para_weight = None
                self.monomer_sequence = None
                n_compounds = None
                self.polymer_lengths = None

        if self.monomer_sequence and self.para_weight:
            warn(
                "Both para_weight and monomer_sequence were given. "
                "The system will be generated using monomer_sequence. "
                "para_weight can only be used when "
                "generating random copolymer sequences. "
                    )
            self.para_weight = None

        if sample_pdi:
            self.sample_from_pdi(
                    mass_dist_type,
                    n_compounds,
                    pdi,
                    Mn,
                    Mw,
                    epsilon
                    )
        elif n_compounds != None: 
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

        init = Initialize(system=self, **kwargs)
        self.system = init.system

    def sample_from_pdi(
            self,
            mass_dist_type,
            n_compounds,
            pdi,
            Mn,
            Mw,
            epsilon
            ):
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
                abs(pdi - (Mw / Mn)) < epsilon
            ), "PDI value does not match Mn and Mw values."
        else:
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
        mass_sampler = mass_distribution_dict["sampler"]
        mass_distribution = mass_distribution_dict["functional_form"]
        samples = np.round(mass_sampler(n_compounds)).astype(int)
        self.polymer_lengths = sorted(list(set(samples)))
        self.n_compounds = [
                list(samples).count(x) for x in self.polymer_lengths
                ]

    def _weibull_k_expression(self, x):
        return (
                (2.0 * x * gamma(2.0 / x)) /
                gamma(1.0 / x) ** 2 - (self.Mw / self.Mn)
                )

    def _weibull_lambda_expression(self, k):
        return self.Mn * k / gamma(1.0 / k)

    def _recover_mass_dist(self, distribution="weibull"):
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
            mass_dict = {
                "sampler": lambda N: np.random.normal(
                    loc=mean, scale=sigma, size=N
                    ),
                "functional_form": lambda x: np.exp(
                    -((x - Mn) ** 2) / (2.0 * sigma)
                    ),
                }
            return mass_dict

        elif distribution == "weibull":
            a = scipy.optimize.root(self._weibull_k_expression, x0=1.0)
            recovered_k = a["x"]
            # get the scale parameter
            recovered_lambda = self._weibull_lambda_expression(recovered_k)
            mass_dict = {
                "sampler": lambda N: recovered_lambda
                * np.random.weibull(recovered_k, size=N),
                "functional_form": lambda x: recovered_k
                / recovered_lambda
                * (x / recovered_lambda) ** (recovered_k - 1)
                * np.exp(-((x / recovered_lambda) ** recovered_k)),
                }
            return mass_dict


class Initialize:
    def __init__(self, system, **kwargs):
        self.system = system
        if system.type == "pack":
            system_init = self.pack()
        elif system.type == "stack":
            system_init = self.stack()
        elif system.type == "custom":
            system_init = self.custom(**kwargs)

        if system.forcefield:
            system_init = self._apply_ff(system_init)

        self.system = system_init

    def pack(self):
        mb_compounds = self._generate_compounds()
        self.L = self._calculate_L() * self.system.expand_factor
        system = mb.packing.fill_box(
            compound=mb_compounds,
            n_compounds=[1 for i in mb_compounds],
            box=[self.L, self.L, self.L],
            overlap=0.2,
            edge=0.9,
            fix_orientation=True,
        )
        system.box = mb.box.Box([self.L, self.L, self.L])
        return system

    def stack(self, separation=1.5):
        mb_compounds = self._generate_compounds()
        self.L = self._calculate_L() * self.system.expand_factor
        system = mb.Compound()
        for idx, comp in enumerate(mb_compounds):
            z_axis_transform(comp)
            comp.translate(np.array([separation,0,0])*idx)
            system.add(comp)
        system.box = mb.box.Box([self.L, self.L, self.L])
        return system

    def custom(self, file_path):
        system = mb.load(file_path)
        mass = sum(
                [ele.element_from_symbol(p.name).mass
                for p in system.particles()]
                )
        self.system.system_mass += mass
        self.L = self._calculate_L()
        system.box = system.get_boundingbox()
        return system

    def _generate_compounds(self):
        if self.system.monomer_sequence:
            sequence = self.system.monomer_sequence
        else:
            sequence = "random"
        random.seed(self.system.seed)
        mb_compounds = []
        for length, n in zip(
                self.system.polymer_lengths,
                self.system.n_compounds
                ):
            for i in range(n):
                polymer, sequence = build_molecule(
                    self.system.molecule,
                    length,
                    sequence,
                    self.system.para_weight
                )
                self.system.molecule_sequences.append(sequence)
                mb_compounds.append(polymer)
                self.system.para += sequence.count("P")
                self.system.meta += sequence.count("M")
            mass = n * sum(
                [ele.element_from_symbol(p.name).mass
                for p in polymer.particles()]
            )
            self.system.system_mass += mass  # amu
        return mb_compounds

    def _calculate_L(self):
        """
        Calcualte the box length needed for entered density
        Right now, assuming cubic box
        Return L in nm (mBuild units)
        """
        M = self.system.system_mass * units["amu_to_g"]  # grams
        L = (M / self.system.density) ** (1 / 3)  # centimeters
        L *= units["cm_to_nm"]  # convert cm to nm
        self.system.target_L = L  # Used during shrink step
        return L

    def _apply_ff(self, untyped_system):
        if self.system.forcefield == "gaff":
            ff_path = f"{FF_DIR}/gaff.xml"
            forcefield = foyer.Forcefield(forcefield_files=ff_path)
        elif self.system.forcefield == "opls":
            forcefield = foyer.Forcefield(name="oplsaa")

        typed_system = forcefield.apply(
            untyped_system,
            assert_dihedral_params=self.system.assert_dihedrals
        )
        if self.system.remove_hydrogens:
            typed_system.strip(
                    [a.atomic_number == 1 for a in typed_system.atoms]
                    )
        return typed_system

class Interface:
    def __init__(
        self,
        slabs,
        ref_distance=None,
        gap=0.1
    ):
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
        x_len = interface.get_boundingbox().Lx
        interface["left"].translate((-x_len - gap, 0, 0))
        
        system_box = mb.box.Box.from_mins_maxs_angles(
                mins=(0, 0, 0),
                maxs = interface.get_boundingbox().lengths,
                angles = (90, 90, 90)
            )
        system_box._Lx += 2 * self.ref_distance * 1.1225
        interface.box = system_box
        # Center in the adjusted box
        interface.translate_to(
                [interface.box.Lx / 2,
                interface.box.Ly / 2,
                interface.box.Lz / 2,]
            )

        ff_path = f"{FF_DIR}/gaff-nosmarts.xml"
        forcefield = foyer.Forcefield(forcefield_files=ff_path)
        self.system = forcefield.apply(interface)

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


def build_molecule(molecule, length, sequence, para_weight, smiles=True):
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
    sequence : str, required
        The monomer sequence to be used when building a polymer.
        Example) "AB" or "AAB"
        If you want a sequence to be generated randomly, use sequence="random"

    para_weight : float, limited to values between 0 and 1
        The relative amount of para configurations compared to meta.
        Passed into random_sequence() to determine the monomer sequence of the
        polymer.
        A 70/30 para to meta system would require a para_weight = 0.70
    smiles : bool, optional, default False
        Set to True if you want to load the molecules from SMILES strings
        If left False, the molecule will be loaded from their .mol2 files

    Notes
    -----
    Any values entered for length and sequence should be compatible.
    This is designed to follow the sequence until it has reached it's
    terminal length. For example::

        A `length=5` and `sequence="PM"` would result in
        `monomer_sequence = "PMPMP".

        The same is true if `length` is shorter than the sequence length
        A `lenght=3` and `sequence="PPMMM"` would result in
        `monomer_sequence = "PPM"`

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

    if sequence == "random":
        monomer_sequence = random_sequence(para_weight, length)
    else:
        n = length // len(sequence)
        monomer_sequence = sequence * n
        monomer_sequence += sequence[:(length - len(monomer_sequence))]

    compound = Polymer()
    if smiles:
        para = mb.load(mol_dict["para_smiles"], smiles=True, backend="rdkit")
        meta = mb.load(mol_dict["meta_smiles"], smiles=True, backend="rdkit")
    else:
        try:
            para = mb.load(os.path.join(COMPOUND_DIR, mol_dict["para_mol2_file"]))
            meta = mb.load(os.path.join(COMPOUND_DIR, mol_dict["meta_mol2_file"]))
        except KeyError:
            print("No mol2 file is available; using the SMILES string instead")
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
