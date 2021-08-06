import json
import os
import random
from warnings import warn

import ele
import foyer
import gsd
import gsd.hoomd
import mbuild as mb
import numpy as np
import scipy.optimize
from foyer import Forcefield
from mbuild.lib.recipes import Polymer
from mbuild.coordinate_transform import z_axis_transform
from scipy.special import gamma

from uli_init.library import COMPOUND_DIR, SYSTEM_DIR, FF_DIR
from uli_init.utils import base_units

units = base_units.base_units()


class System:
    def __init__(
        self,
        density,
        molecule=None,
        n_compounds=None,
        polymer_lengths=None,
        para_weight=None,
        monomer_sequence=None,
        sample_pdi=False,
        pdi=None,
        Mn=None,
        Mw=None,
        seed=24,
    ):
        """
        This class handles the system parameters such as number 
        and type of molecules, length of the polymers, the monomer
        sequence of the polymers, density, and polydispersity.

        Parameters:
        -----------
        density : float, required
            The desired density of the system.
            Requires units of grams per cubic centimeter.
        molecule : str, required
            The type of molecule to be used.
            Supported types are "PEEK" and "PEKK".
            See the .json files in 'library/compounds'
        n_compounds : int or list of int, required
            The number of molecules in the system.
            If using `polymer_lengths` as well `n_compounds`
            and `polymer_lengths` should be equal length.
            If sampling from a PDI, `n_compounds` should be
            a single value.
        polymer_lengths : int or list of int, optional
            The number of monomer units in each molecule
            If using multiple lengths, the number of each
            length will correspond to the `n_compounds`
            parameter. Leave as None if sampling from a PDI.
        para_weight : float (0.0 - 1.0), optional
            Sets the relative amount of para vs meta conformations
            throughout the system. 1.0 is all para, 0.0 is all meta.
            Use this if you want to generate random co-polymers.
            Leave as None if defining the sequence via `monomer_sequence`
        monomer_sequence : str, optional
            Manually defines the co-polymer sequence.
            Example:
            monomer_sequence = "PM" sets an alternative para-meta 
            co-polymer sequence.
        sample_pdi : bool, optional
            If True, the lengths of the polymers will be determined
            by sampling from a distribution.
        pdi : float, optional
            The poly dispersity index used to generate a distribution of
            polymer lengths.
        Mn : float, optional
            The mean molecular length of the poly dispersity distribution
        Mw : float, optional
            The mean molecular mass of the poly dispersity distribution
        seed : int, optional, default=24
            Used to generate random co-polymer sequences
        """
        self.molecule = molecule
        self.para_weight = para_weight
        self.monomer_sequence = monomer_sequence
        self.density = density
        self.seed = seed
        self.system_mass = 0
        self.para = 0
        self.meta = 0
        self.molecule_sequences = []

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
                    )
        elif not sample_pdi and n_compounds != None: 
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

    def sample_from_pdi(
            self,
            mass_dist_type,
            n_compounds,
            pdi,
            Mn,
            Mw,
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
                abs(pdi - (Mw / Mn)) < 1e-7 
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
    def __init__(
            self,
            system,
            system_type,
            forcefield="gaff",
            remove_hydrogens=False,
            assert_dihedrals=True,
            **kwargs):
        """
        system : uli_init.system.System, required
            Contains the parameters for system generation

        system_type : str, required
            The type of system initialization scheme to use for
            generating the morphology.
            Options include:
            'pack': Molecules randomly packed into a box.
            'stack': Polymer chains stacked into layers.
            'crystal': Polymer chains arranged by n x n unit cells.
            'custom': Load a system from a file.

        forcefield : str, optional, default="gaff"
            The type of foyer compatible forcefield to use.
            As of now, only gaff is supported.
        """
        self.system_parms = system
        self.system_type = system_type
        self.forcefield = forcefield
        self.remove_hydrogens = remove_hydrogens
        self.assert_dihedrals = assert_dihedrals
        self.target_box = None

        if self.system_type == "custom":
            system_init = self.custom(**kwargs)
        else:
            self.mb_compounds = self._generate_compounds()
            if self.system_type == "pack":
                system_init = self.pack(**kwargs)
            elif self.system_type == "stack":
                system_init = self.stack(**kwargs)
            elif self.system_type == "crystal":
                system_init = self.crystal(**kwargs)
            else:
                raise ValueError(
                        "Valid system types are:"
                        "'pack'"
                        "'stack'"
                        "'crystal'"
                        "'custom'."
                        "You passed in {system.type}"
                    )

        if self.target_box is None:
            warn("A target box has not been set for this system. "
                 "The default cubic volume (Lx=Ly=Lz) will be used. "
                 "See the `set_target_box()` function to set a non-cubic "
                 "target box."
                 )
            self.target_box = self.set_target_box()

        if self.forcefield:
            self.system = self._apply_ff(system_init)
        else:
            self.system = system_init

    def pack(self, expand_factor=5):
        self.target_box = self.set_target_box()
        pack_box = self.target_box * expand_factor
        system = mb.packing.fill_box(
            compound=self.mb_compounds,
            n_compounds=[1 for i in self.mb_compounds],
            box=list(pack_box),
            overlap=0.2,
            edge=0.9,
            fix_orientation=True,
        )
        return system

    def stack(self, separation=0.7):
        system = mb.Compound()
        for idx, comp in enumerate(self.mb_compounds):
            z_axis_transform(comp)
            comp.translate(np.array([separation,0,0])*idx)
            system.add(comp)

        bounding_box = system.get_boundingbox().lengths
        self.target_box = self.set_target_box(
                z_constraint=bounding_box[2]
                )
        return system

    def crystal(self, a, b, n, vector=[.5, .5, 0]):
        if len(self.mb_compounds) != n*n*2:
            raise ValueError(
                    "The crystal is built as nxn unit cells "
                    "with each unit cell containing 2 molecules. "
                    "The number of molecules should equal 2*n*n"
                    )
        if self.system_parms.para_weight is not None:
            warn("Initializing crystalline systems may not work well "
                 "when generating random co-polymers "
                 "(e.g. overlapping particles). You may want to "
                 "use the `monomer_sequence` as opposed to `para_weight'."
                 )
        next_idx = 0
        crystal = mb.Compound()
        for i in range(n):
            layer = mb.Compound()
            for j in range(n):
                try:
                    comp_1 = self.mb_compounds[next_idx]
                    comp_2 = self.mb_compounds[next_idx+1]
                    z_axis_transform(comp_1)
                    z_axis_transform(comp_2)
                    translate_by = np.array(vector)*(b, a, 0)
                    comp_2.translate(translate_by)
                    unit_cell= mb.Compound(subcompounds=[comp_1, comp_2])
                    unit_cell.translate((0, a*j, 0))
                    layer.add(unit_cell)
                    next_idx += 2
                except IndexError:
                    pass
            layer.translate((b*i, 0, 0))
            crystal.add(layer)

        bounding_box = np.array(crystal.get_boundingbox().lengths)
        self.target_box = self.set_target_box(
                z_constraint=bounding_box[2]
                )
        crystal.box = mb.box.Box(bounding_box*1.05)
        crystal.translate_to(
                (crystal.box.Lx / 2,
                crystal.box.Ly / 2,
                crystal.box.Lz / 2)
                )
        return crystal

    def custom(self, file_path):
        system = mb.load(file_path)
        mass = sum(
                [ele.element_from_symbol(p.name).mass
                for p in system.particles()]
                )
        self.system.system_mass += mass
        return system

    def set_target_box(
            self,
            x_constraint=None,
            y_constraint=None,
            z_constraint=None
            ):
        """
        """
        constraints = np.array([x_constraint, y_constraint, z_constraint])
        if not any([i for i in constraints]): # All edge lengths equal 
            Lx = Ly = Lz = self._calculate_L()
        else:
            fixed_L = constraints[np.where(constraints!=None)]
            #Conv from nm to cm for _calculate_L
            fixed_L /= units["cm_to_nm"]
            L = self._calculate_L(fixed_L = fixed_L)
            constraints[np.where(constraints==None)] = L
            Lx, Ly, Lz = constraints
        return np.array([Lx, Ly, Lz])

    def _calculate_L(self, fixed_L=None):
        """
        """
        M = self.system_parms.system_mass * units["amu_to_g"]  # grams
        vol = (M / self.system_parms.density) # cm^3
        if fixed_L is None: 
            L = vol**(1/3)
        else:
            L = vol / np.prod(fixed_L) 
            if len(fixed_L) == 1: # L is cm^2 
                L = L**(1/2)
        L *= units["cm_to_nm"]  # convert cm to nm
        return L

    def _generate_compounds(self):
        if self.system_parms.monomer_sequence is not None:
            sequence = self.system_parms.monomer_sequence
        else:
            sequence = "random"
        random.seed(self.system_parms.seed)
        mb_compounds = []
        for length, n in zip(
                self.system_parms.polymer_lengths,
                self.system_parms.n_compounds
                ):
            for i in range(n):
                polymer, mol_sequence = build_molecule(
                    self.system_parms.molecule,
                    length,
                    sequence,
                    self.system_parms.para_weight
                )
                self.system_parms.molecule_sequences.append(mol_sequence)
                mb_compounds.append(polymer)
                self.system_parms.para += sequence.count("P")
                self.system_parms.meta += sequence.count("M")
            mass = n * sum(
                [ele.element_from_symbol(p.name).mass
                for p in polymer.particles()]
            )
            self.system_parms.system_mass += mass  # amu
        return mb_compounds

    def _apply_ff(self, untyped_system):
        if self.forcefield == "gaff":
            ff_path = f"{FF_DIR}/gaff.xml"
            forcefield = foyer.Forcefield(forcefield_files=ff_path)
        elif self.forcefield == "opls":
            forcefield = foyer.Forcefield(name="oplsaa")

        typed_system = forcefield.apply(
            untyped_system,
            assert_dihedral_params=self.assert_dihedrals
        )
        if self.remove_hydrogens:
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
        self.system_type = "interface"
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


def build_molecule(molecule, length, sequence, para_weight, smiles=False):
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
        monomer_sequence = "".join(random_sequence(para_weight, length))
    else:
        n = length // len(sequence)
        monomer_sequence = sequence * n
        monomer_sequence += sequence[:(length - len(monomer_sequence))]
        monomer_sequence = "".join(monomer_sequence)

    compound = Polymer()
    if smiles:
        para = mb.load(mol_dict["para_smiles"], smiles=True, backend="rdkit")
        meta = mb.load(mol_dict["meta_smiles"], smiles=True, backend="rdkit")
    else:
        try:
            para = mb.load(os.path.join(COMPOUND_DIR, mol_dict["para_file"]))
            meta = mb.load(os.path.join(COMPOUND_DIR, mol_dict["meta_file"]))
        except KeyError:
            print("No file is available for this compound")

    if len(set(monomer_sequence)) == 2: # Copolymer
        compound.add_monomer(meta, 
                mol_dict["meta_bond_indices"],
                mol_dict["bond_distance"],
                mol_dict["bond_orientation"],
                replace=True
            )
        compound.add_monomer(para,
                mol_dict["para_bond_indices"],
                mol_dict["bond_distance"],
                mol_dict["bond_orientation"],
                replace=True
            )
    else:
        if monomer_sequence[0] == "P": # Only para
            compound.add_monomer(para,
                    mol_dict["para_bond_indices"],
                    mol_dict["bond_distance"],
                    mol_dict["bond_orientation"],
                    replace=True
                )
        elif monomer_sequence[0] == "M": # Only meta
            compound.add_monomer(meta,
                    mol_dict["meta_bond_indices"],
                    mol_dict["bond_distance"],
                    mol_dict["bond_orientation"],
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
