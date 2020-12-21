import json
import os
import random
import numpy as np
from uli_init.utils import smiles_utils, polysmiles, base_units
from uli_init.compounds import COMPOUND_DIR
import hoomd
import mbuild as mb
from mbuild.formats.hoomd_simulation import create_hoomd_simulation
import foyer
from foyer import Forcefield
import ele
import operator
from collections import namedtuple
import scipy.optimize
from scipy.special import gamma

units = base_units.base_units()


class Simulation():
    def __init__(self,
                 system,
                 target_box=None,
                 r_cut = 1.2,
                 e_factor = 0.5,
                 tau = 0.1,
                 dt = 0.0001,
                 auto_scale = True,
                 ref_units = None,
                 mode = "gpu",
                 gsd_write = 1e4,
                 log_write = 1e3,
                 seed = 42
                 ):

        self.system = system
        self.system_pmd = system.system_pmd # Parmed structure
        self.r_cut = r_cut
        self.e_factor = e_factor
        self.tau = tau
        self.dt = dt
        self.auto_scale = auto_scale
        self.ref_units = ref_units
        self.mode = mode
        self.gsd_write = gsd_write
        self.log_write = log_write
        self.seed = seed

        if ref_units and not auto_scale:
            self.ref_energy = ref_units['energy']
            self.ref_distance = ref_units['distance']
            self.ref_mass = ref_units['mass']

        elif auto_scale and not ref_units: # Pulled from mBuild hoomd_simulation.py
            self.ref_mass = max([atom.mass for atom in self.system_pmd.atoms])
            pair_coeffs = list(set((atom.type,
                                    atom.epsilon,
                                    atom.sigma) for atom in self.system_pmd.atoms))
            self.ref_energy = max(pair_coeffs, key=operator.itemgetter(1))[1]
            self.ref_distance = max(pair_coeffs, key=operator.itemgetter(2))[2]

        self.reduced_target_L = self.system.target_L / self.ref_distance # nm
        self.reduced_init_L = self.system_pmd.box[0] / self.ref_distance # angstroms

        #TODO: Use target_box to generate non-cubic simulation volumes
        if target_box:
            self.target_box = target_box
        else:
            self.target_box = [self.reduced_target_L]*3

        self.log_quantities = [
        "temperature",
        "pressure",
        "volume",
        "potential_energy",
        "kinetic_energy",
        "pair_lj_energy",
        "bond_harmonic_energy",
        "angle_harmonic_energy"
        ]
        

    def quench(self, kT, n_steps, shrink_kT=10, shrink_steps=1e6):
        '''
        '''
        # Get hoomd stuff set:
        create_hoomd_simulation(self.system_pmd, self.ref_distance,
                                self.ref_mass, self.ref_energy,
                                self.r_cut, self.auto_scale)
        _all = hoomd.group.all()
        hoomd.md.integrate.mode_standard(dt=self.dt)
        integrator = hoomd.md.integrate.nvt(group=_all, kT=shrink_kT, tau=self.tau) # shrink temp
        integrator.randomize_velocities(seed=self.seed)
        
        # Set up shrinking box_updater:
        shrink_gsd = hoomd.dump.gsd("traj-shrink.gsd",
                       period=self.gsd_write, group=_all, phase=0, overwrite=True)
        x_variant = hoomd.variant.linear_interp([(0, self.reduced_init_L),
                                                 (shrink_steps, self.target_box[0]*10)])
        y_variant = hoomd.variant.linear_interp([(0, self.reduced_init_L),
                                                 (shrink_steps, self.target_box[1]*10)])
        z_variant = hoomd.variant.linear_interp([(0, self.reduced_init_L),
                                                 (shrink_steps, self.target_box[2]*10)])
        box_updater = hoomd.update.box_resize(Lx = x_variant, Ly = y_variant, Lz = z_variant)

        # Run the shrink portion of simulation
        hoomd.run_upto(shrink_steps)
        shrink_gsd.disable()
        box_updater.disable()

        # Set up new gsd and log dumps for actual simulation
        hoomd.dump.gsd("sim_traj.gsd",
                       period=self.gsd_write,
                       group=_all,
                       phase=0,
                       overwrite=True)
        hoomd.analyze.log("sim_traj.log",
                          period=self.log_write,
                          quantities = self.log_quantities,
                          header_prefix="#",
                          overwrite=True, phase=0)
        # Run the primary simulation
        integrator.set_params(kT=kT)
        integrator.randomize_velocities(seed=self.seed)
        hoomd.run(n_steps)


    def anneal(self,
              kT_init=None,
              kT_final=None,
              step_sequence=None,
              schedule=None,
              shrink_kT=10,
              shrink_steps=1e6
              ):

        if not schedule:
            temps = np.linspace(kT_init, kT_final, len(step_sequence))
            temps = [np.round(t, 1) for t in temps]
            schedule = dict(zip(temps, step_sequence))

        # Get hoomd stuff set:
        create_hoomd_simulation(self.system_pmd, self.ref_distance,
                                self.ref_mass, self.ref_energy,
                                self.r_cut, self.auto_scale)
        _all = hoomd.group.all()
        hoomd.md.integrate.mode_standard(dt=self.dt)
        integrator = hoomd.md.integrate.nvt(group=_all, kT=shrink_kT, tau=self.tau) # shrink temp
        integrator.randomize_velocities(seed=self.seed)
        
        # Set up shrinking box_updater:
        shrink_gsd = hoomd.dump.gsd("traj-shrink.gsd",
                       period=self.gsd_write, group=_all, phase=0, overwrite=True)
        x_variant = hoomd.variant.linear_interp([(0, self.reduced_init_L),
                                                 (shrink_steps, self.target_box[0]*10)])
        y_variant = hoomd.variant.linear_interp([(0, self.reduced_init_L),
                                                 (shrink_steps, self.target_box[1]*10)])
        z_variant = hoomd.variant.linear_interp([(0, self.reduced_init_L),
                                                 (shrink_steps, self.target_box[2]*10)])
        box_updater = hoomd.update.box_resize(Lx = x_variant, Ly = y_variant, Lz = z_variant)

        # Set up new log and gsd files for simulation:
        anneal_gsd = hoomd.dump.gsd("traj-anneal.gsd",
                                   period=self.gsd_write,
                                   group=_all,
                                   phase=0,
                                   overwrite=True)
        hoomd.analyze.log("sim_traj.log",
                          period=self.log_write,
                          quantities = self.log_quantities,
                          header_prefix="#",
                          overwrite=True, phase=0)
        # Start annealing steps:
        last_time_step = shrink_steps
        for kT in schedule:
            print('Running @ Temp = {}'.format(kT))
            n_steps = schedule[kT]
            print('Running for {} steps'.format(n_steps))
            integrator.set_params(kT=kT)
            integrator.randomize_velocities(seed=self.seed)
            hoomd.run(last_time_step + n_steps)
            last_time_step += n_steps
            print()


class System():
    
    def __init__(self,
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
                 mass_dist_type='weibull',
                 remove_hydrogens=False,
                 assert_dihedrals=True,
                 seed=24
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
        self.para = 0 # keep track for now to check things are working, maybe keep?
        self.meta = 0
        
        if sample_pdi:
            if isinstance(n_compounds, int):
                self.n_compounds = n_compounds
            elif isinstance(n_compounds, list) and len(n_compounds) == 1:
                self.n_compounds = n_compounds[0]
            elif isinstance(n_compounds, list) and len(n_compounds) != 1:
                raise TypeError('n_compounds should be of length 1 when sample_pdi is True.')
            pdi_arg_sum = sum([x is not None for x in [pdi, Mn, Mw]])
            assert pdi_arg_sum >= 2, 'At least two of [pdi, Mn, Mw] must be given.'
            if pdi_arg_sum == 3:
                #special case, make sure that pdi = Mw / Mn
                assert abs(pdi - (Mw/Mn)) < self.epsilon, 'PDI value does not match Mn and Mw values.'
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
                    
            # this returns a numpy.random callable set up with recovered parameters
            mass_distribution_dict = self._recover_mass_dist(mass_dist_type)
            self.mass_sampler = mass_distribution_dict['sampler']
            self.mass_distribution = mass_distribution_dict['functional_form']
            # TODO: make sure we don't sample any negative weights
            samples = np.round(self.mass_sampler(n_compounds)
                              ).astype(int)
            # get all unique lengths in increasing order
            self.polymer_lengths = sorted(list(set(samples)))
            # get count of each length
            self.n_compounds = [list(samples).count(x) for x in self.polymer_lengths]
            print(f'polymer_lengths: {self.polymer_lengths}, n_compounds: {self.n_compounds}')
            
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
        
        self.system_mb = self._pack() # mBuild object before applying FF
        if self.forcefield:
            self.system_pmd = self._type_system() # parmed object after applying FF
            self.system_pmd.save('init.pdb', overwrite=True)
        
    def _weibull_k_expression(self, x):
        return (2. * x * gamma(2./x)) / gamma(1./x)**2 - (self.Mw / self.Mn)
    
    def _weibull_lambda_expression(self, k):
        return self.Mn * k / gamma(1./k)
    
    def _recover_mass_dist(self, distribution='Gaussian'):
        '''This function takes in two of the three quantities [Mn, Mw, PDI],
           and fits either a Gaussian or Weibull distribution of molar masses to them.'''
        if distribution.lower() != 'gaussian' and distribution.lower() != 'weibull':
            raise(ValueError('Molar mass distribution must be either "gaussian" or "weibull".'))
        if distribution.lower() == 'gaussian':
            mean = self.Mn
            sigma = self.Mn * (self.Mw - self.Mn)
            return {'sampler': lambda N: np.random.normal(loc=mean, scale=sigma, size=N),
                    'functional_form': lambda x: np.exp(-(x-Mn)**2 / (2. * sigma))}
        elif distribution.lower() == 'weibull':
            # get the shape parameter
            a = scipy.optimize.root(self._weibull_k_expression, x0=1.)
            recovered_k = a['x']
            # get the scale parameter
            recovered_lambda = self._weibull_lambda_expression(recovered_k)
            return {'sampler': lambda N: recovered_lambda * np.random.weibull(recovered_k, size=N),
                    'functional_form': lambda x: recovered_k / recovered_lambda * (x / recovered_lambda) ** (recovered_k - 1) * np.exp(- (x / recovered_lambda) ** recovered_k)}
        
    def _pack(self, box_expand_factor=5):
        random.seed(self.seed)
        mb_compounds = []
        for _length, _n in zip(self.polymer_lengths, self.n_compounds):
            for i in range(_n):
                polymer, sequence = build_molecule(self.molecule, _length,
                                        self.para_weight)

                mb_compounds.append(polymer)
                self.para += sequence.count('para')
                self.meta += sequence.count('meta')
            mass = _n * np.sum(ele.element_from_symbol(p.name).mass for p in polymer.particles())
            self.system_mass += mass # amu
        
        # Figure out correct box dimensions and expand the box to make the PACKMOL step faster
        # Will shrink down to accurate L during simulation
        L = self._calculate_L() * box_expand_factor
        system = mb.packing.fill_box(
            compound = mb_compounds,
            n_compounds = [1 for i in mb_compounds],
            box=[L, L, L],
            overlap=0.2,
            edge=0.9,
            fix_orientation=True)
        system.Box = mb.box.Box([L, L, L])
        return system
    
    
    def _type_system(self):
        if self.forcefield == 'gaff':
            forcefield = foyer.forcefields.load_GAFF()
        elif self.forcefield == 'opls':
            forcefield = foyer.Forcefield(name='oplsaa')
        
        typed_system = forcefield.apply(self.system_mb,
                                       assert_dihedral_params=self.assert_dihedrals)
        if self.remove_hydrogens:
            typed_system.strip([a.atomic_number == 1 for a in typed_system.atoms])
        return typed_system
    
    def _calculate_L(self):
        '''
        Calcualte the box length needed for entered density
        Right now, assuming cubic box
        Return L in nm (mBuild units)
        '''
        M = self.system_mass * units["amu_to_g"] # grams
        L = (M / self.density)**(1/3) # centimeters
        L *= units['cm_to_nm'] # convert cm to nm
        self.target_L = L # Used during shrink step
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
    compound : mBuild.Compound
        An instance of the single polymer created
    sequence : list
        List of the configuration sequence of the finished compound
    '''
    f = open('{}/{}.json'.format(COMPOUND_DIR, molecule))
    mol_dict = json.load(f)
    f.close()
    monomer_sequence = random_sequence(para_weight, length)
    molecule_string = '{}'

    for idx, config in enumerate(monomer_sequence):
        if idx == 0: # append template, but not brackets
            monomer_string = mol_dict['{}_template'.format(config)]
            if molecule == 'PEEK': # Change oxygen type on the terminal end of the polymer; needs its hydrogen.
                monomer_string = "O"+monomer_string[1:]
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
