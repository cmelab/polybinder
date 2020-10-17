import json
import os
import random
import numpy as np
from utils import smiles_utils
from utils import polysmiles
import hoomd
import mbuild as mb
from mbuild.formats.hoomd_simulation import create_hoomd_simulation
import foyer
from foyer import Forcefield
import py3Dmol
import ele

class Simulation():
    def __init__(self,
                 system,
                 kwargs,
                 target_box,
                 r_cut = 1.2,
                 e_factor = 0.5,
                 tau = 1,
                 dt = 0.001,
                 auto_scale = True,
                 ref_units = None,
                 mode = "gpu",
                 gsd_write = 1e4,
                 log_write = 1e3
                 ):
        self.system = system # Parmed structure
        self.r_cut = r_cut
        self.e_factor = e_factor
        self.tau = tau
        self.dt = dt
        self.auto_scale = auto_cale
        self.ref_units = ref_units
        self.target_box = target_box
        self.mode = mode
        self.gsd_write = gsd_write
        self.log_write = log_write

        if ref_units and not auto_scale:
            self.ref_energy = ref_units['energy']
            self.ref_distance = ref_units['distance']
            self.ref_mass = ref_units['mass']
        elif auto_scale and not ref_units:
            self.ref_energy = 1
            self.ref_distance = 1
            self.ref_mass = 1

    def quench(self, kT, n_steps):

        # Get hoomd stuff set
        create_hoomd_simulation(self.system, self.ref_distance,
                                self.ref_mass, self.ref_energy,
                                self.r_cut, self.auto_scale)
        _all = hoomd.group.all()
        hoomd.md.integrate.mode_standard(dt=self.dt)

        # Run shrinking step
        integrator = hoomd.md.integrate.nvt(group=_all, kT=10, tau=self.tau) # shrink temp
        integrator.randomize_velocities(seed=42)
        hoomd.dump.gsd("trajectories/traj-shrink.gsd",
                       period=self.gsd_write, group=_all, phase=0, overwrite=True)
        x_variant = hoomd.variant.linear_interp([(0, self.system.box[0]),
                                                 (1e6, self.target_box[0]*10)])
        y_variant = hoomd.variant.linear_interp([(0, self.system.box[1]),
                                                 (1e6, self.target_box[1]*10)])
        z_variant = hoomd.variant.linear_interp([(0, self.system.box[2]),
                                                 (1e6, self.target_box[2]*10)])
        box_resize = hoomd.update.box_resize(Lx = x_variant, Ly = y_variant, Lz = z_variant)
        hoomd.run_upto(1e4)
        hoomd.dump.gsd.disable()
        box_resize.disable()

        # Run primary simulation
        hoomd.dump.gsd("trajectories/sim_traj.gsd",
                       period=self.gsd_write, group=_all, phase=0, overwrite=True)
        hoomd.analyze.log("logs/sim_traj.log",
                          period=self.log_write, group=_all, header_prefix="#",
                          overwrite=True, phase=0)

        integrator.set_params(kT=kT)
        integrator.randomize_velocities(seed=42)
        hoomd.run(n_steps)


    def anneal(self,
              kT_init,
              kT_final,
              step_sequence):
        pass

class System():
    def __init__(self,
                 molecule,
                 para_weight,
                 density,
                 n_compounds,
                 polymer_lengths,
                 forcefield=None,
                 pdi=None,
                 M_n=None,
                 remove_hydrogens=False
                ):
        self.molecule = molecule
        self.para_weight = para_weight
        self.density = density
        self.remove_hydrogens = remove_hydrogens
        self.pdi = pdi
        self.forcefield = forcefield
        self.system_mass = 0
        self.para = 0 # keep track for now to check things are working, maybe keep?
        self.meta = 0

        if self.pdi:
            pass
            '''
            Here, call a function that samples from some distribution
            pass in pdi, n_compounds, M_n?
            self.polymer_lengths and self.n_compounds defined from that function
            '''
        else: # Do some validation, get things in the correct data types
            if not isinstance(n_compounds, list):
                self.n_compounds = [n_compounds]
            else:
                self.n_compounds = n_compounds

            if not isinstance(polymer_lengths, list):
                self.polymer_lengths = [polymer_lengths]
            else:
                self.polymer_lengths = polymer_lengths

        if len(self.n_compounds) != len(self.polymer_lengths):
            raise ValueError('n_compounds and polymer_lengths should be equal length')

        self.system = self.pack() # mBuild object before applying FF
        if self.forcefield:
            self.system = self.type_system() # parmed object after applying FF


    def pack(self, box_expand_factor=5):
        mb_compounds = []
        for _length, _n in zip(self.polymer_lengths, self.n_compounds):
            for i in range(_n):
                polymer, sequence = build_molecule(self.molecule, _length,
                                        self.para_weight)

                mb_compounds.append(polymer)
                self.para += sequence.count('para')
                self.meta += sequence.count('meta')
            mass = _n * np.sum(ele.element_from_symbol(p.name).mass for p in polymer.particles())
            self.system_mass += mass

        # Figure out correct box dimensions and expand the box to make the PACKMOL step faster
        # Will shrink down to accurate L during simulation
        L = self._calculate_L() * box_expand_factor

        system = mb.packing.fill_box(
            compound = mb_compounds,
            n_compounds = [1 for i in mb_compounds],
            box=[L, L, L],
            edge=0.5,
            fix_orientation=True)
        return system


    def type_system(self):
        if self.forcefield == 'gaff':
            forcefield = foyer.forcefields.load_GAFF()
        elif self.forcefield == 'opls':
            forcefield = foyer.Forcefield(name='oplsaa')

        typed_system = forcefield.apply(self.system)
        if self.remove_hydrogens: # not sure how to do this with Parmed yet
            removed_hydrogen_count = 0 # subtract from self.mass
            pass
        return typed_system

    def _calculate_L(self):
        # Conversion from (amu/(g/cm^3)) to ang
        L = (self.system_mass / self.density) ** (1/3) * 1.841763
        L /= 10 # convert ang to nm
        return L


def build_molecule(molecule, length, para_weight):
    '''
    `build_molecule` uses SMILES strings to build up a polymer from monomers.
    The configuration of each monomer is determined by para_weight and the
    random_sequence() function.
    Uses DeepSMILES behind the scenes to build up SMILES string for a polymer.

    Parameters
    ----------
    molecule : str
        The monomer molecule to be used to build up the polymer.
        Available options are limited  to the .json files in the compounds directory
        Use the molecule name as seen in the .json file without including .json
    length : int
        The number of monomer units in the final polymer molecule
    para_weight : float, limited to values between 0 and 1
        The relative amount of para configurations compared to meta.
        Passed into random_sequence() to determine the monomer sequence of the polymer.
        A 70/30 para to meta system would require a para_weight = 0.70

    Returns
    -------
    molecule_string_smiles : str
        The complete SMILES string of the polymer molecule
    '''
    f = open('compounds/{}.json'.format(molecule))
    mol_dict = json.load(f)
    f.close()
    monomer_sequence = random_sequence(para_weight, length)
    molecule_string = '{}'

    for idx, config in enumerate(monomer_sequence):
        if idx == 0: # append template, but not brackets
            monomer_string = mol_dict['{}_template'.format(config)]
            molecule_string = molecule_string.format(monomer_string)
            if len(monomer_sequence) == 1:
                molecule_string = molecule_string.replace('{}', '')
                continue

        elif idx == length - 1: # Don't use template for last iteration
            brackets = polysmiles.count_brackets(mol_dict['{}_deep_smiles'.format(config)])
            monomer_string = mol_dict['{}_deep_smiles'.format(config)]
            molecule_string = molecule_string.format(monomer_string, brackets)

        else: # Continue using template plus brackets
            brackets = polysmiles.count_brackets(mol_dict['{}_deep_smiles'.format(config)])
            monomer_string = mol_dict['{}_template'.format(config)]
            molecule_string = molecule_string.format(monomer_string, brackets)

    molecule_string_smiles = smiles_utils.convert_smiles(deep = molecule_string)
    compound = mb.load(molecule_string_smiles, smiles=True)
    return compound, monomer_sequence


def random_sequence(para_weight, length):
    '''
    random_sequence returns a list containing a random sequence of strings 'para' and 'meta'.
    This is used by build_molecule() to create a complete SMILES string of a molecule.

    Parameters:
    -----------
    para_weight : float, limited to values between 0 and 1
        The relative amount of para configurations compared to meta.
        Defined in build_molecule()
    length : int
        The number of elements in the random sequence.
        Defined in build_molecule()
    '''
    meta_weight = 1 - para_weight
    options = ['para', 'meta']
    probability = [para_weight, meta_weight]
    sequence = random.choices(options, weights=probability, k=length)
    return sequence
