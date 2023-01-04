import json
import os
import random
from warnings import warn
import pickle

import ele
import foyer
from gmso.external import from_mbuild, to_parmed
import gsd
import gsd.hoomd
import mbuild as mb
from mbuild.lib.recipes import Polymer
from mbuild.coordinate_transform import z_axis_transform
from mbuild.formats.hoomd_snapshot import to_hoomdsnapshot
import numpy as np
import parmed
import scipy.optimize
from scipy.special import gamma

from polybinder.library import COMPOUND_DIR, SYSTEM_DIR, FF_DIR
from polybinder.utils import base_units

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
            Supported types are "PEEK", "PEKK", and "PPS".
            See the .json files in 'library/compounds'
        n_compounds : int or list of int, required
            The number of molecules in the system.
            If using `polymer_lengths` as well `n_compounds`
            then `polymer_lengths` should be equal length.
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
        self.n_compounds = n_compounds
        self.polymer_lengths = polymer_lengths
        self.para_weight = para_weight
        self.monomer_sequence = monomer_sequence
        self.density = density
        self.seed = seed
        self.para = 0
        self.meta = 0
        self.molecule_sequences = []

        if self.molecule != "PEKK":
            if para_weight not in [1.0, None] or "M" in str(monomer_sequence):
                raise ValueError(
                        "The meta backbone bonding configuraiton "
                        "is only supported for the PEKK molecule. "
                        f"Since you are using {self.molecule} either use "
                        "para_weight = 1.0 or monomer_sequence = 'P'."
                )

        if self.monomer_sequence and self.para_weight:
            warn(
                "Both para_weight and monomer_sequence were given. "
                "The system will be generated using monomer_sequence. "
                "para_weight should only be used when "
                "generating random copolymer sequences. "
            )
            self.para_weight = None

        if sample_pdi:
            self.sample_from_pdi(
                    n_compounds, pdi, Mn, Mw,
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
                        "n_compounds and polymer_lengths must be equal length"
                )

    def sample_from_pdi(
            self,
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
        # this returns a numpy.random callable set up with recovered params
        mass_distribution_dict = self._recover_mass_dist()
        mass_sampler = mass_distribution_dict["sampler"]
        mass_distribution = mass_distribution_dict["functional_form"]
        samples = np.round(mass_sampler(n_compounds)).astype(int)
        if len(samples[samples == 0]) > 0:
            samples[samples == 0] = 1
            warn("Polymer lengths of 0 were returned from sampling. "
                "They were replaced a length of 1"
            )
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

    def _recover_mass_dist(self):
        """This function takes in two of the three quantities [Mn, Mw, PDI],
        and fits a Weibull distribution of molar masses to them.

        """
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


class Initializer:
    def __init__(
            self,
            system,
            forcefield=None,
            charges=None,
            remove_hydrogens=False,
            save_parmed=True,
            parmed_dir="pmd_structures",
    ):
        """
        Given a set of system parameters, this class handles
        the system initialzation steps.

        In this case, system initialization includes:
            1. The molecular arrangement
            2. Setting system volume
            3. Removal of hydrogens if desired
            4. Application of a forcefield

        Parameters:
        ----------
        system : polybinder.system.System, required
            Contains the parameters for system generation
        system_type : str, required
            The type of system initialization scheme to use for
            generating the starting morphology.
            Options include:
            'pack': Molecules randomly packed into a box.
            'stack': Polymer chains stacked into layers.
            'crystal': Polymer chains arranged by n x n unit cells.
        forcefield : str, optional, default="gaff"
            The type of foyer compatible forcefield to use.
            As of now, only gaff is supported.
        charges : str, defualt = None
            Specifies the charges dict to use. See the .JSON file for the
            molecule of interest.
        remove_hydrogens : bool, optional, default=False
            If True, hydrogen atoms are removed from the system.
        save_parmed : bool, optional, default=True
            If True, the parmed structure is saved to the file.
        parmed_dir: str, optional, default="pmd_structures"
            Directory for saving the parmed structures.

        """
        self.system_parms = system
        self.forcefield = forcefield
        self.remove_hydrogens = remove_hydrogens
        self.system_mass = 0
        self.target_box = None
        self.charges = charges
        self.system_type = None
        self.save_parmed = save_parmed
        self.untyped_system = None
        if self.save_parmed:
            os.makedirs(parmed_dir, exist_ok=True)
            self.pmd_pickle_path = os.path.join(
                    parmed_dir,'parmed.pickle'
            )
        else:
            self.pmd_pickle_path = None

        self.mb_compounds = self._generate_compounds()
        self.cg_compounds = [] 

    def pack(self, expand_factor=7):
        """Uses PACKMOL via mBuild to randomlly fill a box
        with the generated polymer chains.

        Parameters:
        -----------
        expand_factor : int, required, default = 7
            How much to multiply each box edge length by
            before passing to PACKMOL to fill.

        """
        if self.cg_compounds:
            compounds = self.cg_compounds
        else:
            compounds = self.mb_compounds
        self.set_target_box()
        pack_box = self.target_box * expand_factor
        system = mb.packing.fill_box(
            compound=compounds,
            n_compounds=[1 for i in compounds],
            box=list(pack_box),
            overlap=0.2,
            edge=0.9,
            fix_orientation=True,
        )
        if self.forcefield or self.cg_compounds:
            self._load_parmed_structure(untyped_system=system)
            if self.remove_hydrogens:
                self._remove_hydrogens()
        else:
            self.system = system
        self.system_type = "pack"

    def stack(self, separation=0.7):
        """This method organizes the polymer chains in a single layer.
        This approach may be useful for simulating a small number of
        chains at a low density to see the chain-specific behavior.

        Parameters:
        -----------
        separation : float, required, default=0.7
            The distances (nm) between individually stacked chains.

        """
        if self.cg_compounds:
            compounds = self.cg_compounds
        else:
            compounds = self.mb_compounds
        self.set_target_box()
        system = mb.Compound()
        for idx, comp in enumerate(compounds):
            z_axis_transform(comp)
            comp.translate(np.array([0,0,separation])*idx)
            system.add(comp)

        system.box = mb.box.Box(self.target_box)
        # Center the chains in the box
        system.translate_to((
            system.box.Lx / 2,
            system.box.Ly / 2,
            system.box.Lz / 2
        ))

        if self.forcefield or self.cg_compounds:
            self._load_parmed_structure(untyped_system=system)
            if self.remove_hydrogens:
                self._remove_hydrogens()
        else:
            self.system = system
        self.system_type = "stack"

    def crystal(self, a, b, n, vector=[.5, .5, 0], z_adjust=1.0):
        """Creates a system of n x n repeating unit cells
        where each unit cell contains 2 molecules.

        Parameters:
        -----------
        a : float, required
            Sets the distance between repeat units in the a direction
        b : float, required
            Sets the distance between repeat units in the b direction
        n : int, required
            The number of times to repeat a unit cell in both of
            the a and b direction.
        vector : np.array, optional, default = [.5, .5, 0]
            The vector to translate the 2nd molecule by
        z_adjust : float, optional, default = 1.0
            By default, the length of the bounding box along the
            length of the chains is used to set the z_constraint in
            `set_target_box`. z_adjust can be used to adjust the target
            length of the simulation volume in the z direction. It is
            used as a multiplier of the bounding box z length.

        """
        if self.cg_compounds:
            compounds = self.cg_compounds
        else:
            compounds = self.mb_compounds
        if len(compounds) != n*n*2:
            raise ValueError(
                    "The crystal is built as n x n unit cells "
                    "with each unit cell containing 2 molecules. "
                    "The number of molecules should equal 2*n*n"
            )
        if self.system_parms.para_weight not in [None, 1.0, 0.0]:
            warn("Initializing crystalline systems may not work well "
                 "when usingg random co-polymers "
                 "(e.g. overlapping particles). You may want to "
                 "use `monomer_sequence` as opposed to `para_weight'."
            )
        next_idx = 0
        crystal = mb.Compound()
        for i in range(n):
            layer = mb.Compound()
            for j in range(n):
                try:
                    comp_1 = compounds[next_idx]
                    comp_2 = compounds[next_idx+1]
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
        target_z = bounding_box[-1] * z_adjust
        bounding_box[0] *= 1.04
        bounding_box[1] *= 1.04
        bounding_box[2] *= 1.01
        self.set_target_box(z_constraint=target_z)
        crystal.box = mb.box.Box(bounding_box)
        # Center in the box
        crystal.translate_to(
                (crystal.box.Lx / 2, crystal.box.Ly / 2, crystal.box.Lz / 2)
        )

        if self.forcefield or self.cg_compounds:
            self._load_parmed_structure(untyped_system=crystal)
            if self.remove_hydrogens:
                self._remove_hydrogens()
        else:
            self.system = crystal
        self.system_type = "crystal"
    
    def coarse_grain_system(
            self, use_monomers=False, use_components=False, bead_mapping=None
    ):
        import polybinderCG.mbuild_cg as mbcg

        if self.forcefield:
            raise ValueError(
                    "Using foyer forcefield files are not "
                    "supported with coarse-grained systems. "
                    "See polybinder.simulate.Simulation() "
                    "for using coarse model forcefields. "
            )

        for idx, comp in enumerate(self.mb_compounds):
            cg_comp = mbcg.System(
                    mb_compound=comp, molecule=self.system_parms.molecule
            )
            if use_monomers:
                for mol in cg_comp.molecules:
                    mol.sequence = self.system_parms.molecule_sequences[idx]
                    mol.assign_types()
            elif use_components:
                for mon in cg_comp.monomers():
                    mon.generate_components(index_mapping=bead_mapping)
            self.cg_compounds.append(
                    cg_comp.save(
                        use_monomers=use_monomers, use_components=use_components
                    )
            )

    def set_target_box(
            self,
            x_constraint=None,
            y_constraint=None,
            z_constraint=None
    ):
        """Set the target volume of the system during
        the initial shrink step.
        If no constraints are set, the target box is cubic.
        Setting constraints will hold certain box vectors
        constant and adjust others to match the target density.

        Parameters:
        -----------
        x_constraint : float, optional, defualt=None
            Fixes the box length along the x axis
        y_constraint : float, optional, default=None
            Fixes the box length along the y axis
        z_constraint : float, optional, default=None
            Fixes the box length along the z axis
        """
        if not any([x_constraint, y_constraint, z_constraint]):
            Lx = Ly = Lz = self._calculate_L()
        else:
            constraints = np.array([x_constraint, y_constraint, z_constraint])
            fixed_L = constraints[np.where(constraints!=None)]
            #Conv from nm to cm for _calculate_L
            fixed_L /= units["cm_to_nm"]
            L = self._calculate_L(fixed_L = fixed_L)
            constraints[np.where(constraints==None)] = L
            Lx, Ly, Lz = constraints
        self.target_box = np.array([Lx, Ly, Lz])

    @property
    def net_charge(self):
        if isinstance(self.system, parmed.structure.Structure):
            return sum([a.charge for a in self.system.atoms])
        elif isinstance(self.system, mb.Compound):
            warn("A forcefield wasn't applied to the system. No charges exist")
            return 0

    def _calculate_L(self, fixed_L=None):
        """Calculates the required box length(s) given the
        mass of a sytem and the target density.

        Box edge length constraints can be set by set_target_box().
        If constraints are set, this will solve for the required
        lengths of the remaining non-constrained edges to match
        the target density.
        """
        M = self.system_mass * units["amu_to_g"]  # grams
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
        """Generates a list of mbuild.Compound objects
        from the number and lengths given by the
        system parameters.
        This function also updates the mass of the system
        as compounds are generated and records the
        monomer sequence of each molecule.
        """
        if self.system_parms.monomer_sequence is not None:
            sequence = self.system_parms.monomer_sequence
        else:
            sequence = "random"
        random.seed(self.system_parms.seed)
        mb_compounds = []
        self.residues = []
        for length, n in zip(
                self.system_parms.polymer_lengths, self.system_parms.n_compounds
        ):
            if sequence != "random":
                polymer, mol_sequence = build_molecule(
                    molecule=self.system_parms.molecule,
                    length=length,
                    sequence=sequence,
                    para_weight=self.system_parms.para_weight,
                    charges=self.charges
                )
                polymer.name = f"{mol_sequence}_{length}mer"
                mb_compounds.append(polymer)
                self.residues.extend([polymer.name]*n)
                self.system_parms.molecule_sequences.append(mol_sequence)
                for i in range(n - 1):
                    _clone = mb.clone(polymer)
                    mb_compounds.append(_clone)
                    self.system_parms.molecule_sequences.append(mol_sequence)
                self.system_parms.para += (mol_sequence.count("P") * n)
                self.system_parms.meta += (mol_sequence.count("M") * n)

            elif sequence == "random":
                for i in range(n):
                    polymer, mol_sequence = build_molecule(
                        molecule=self.system_parms.molecule,
                        length=length,
                        sequence=sequence,
                        para_weight=self.system_parms.para_weight,
                        charges=self.charges
                    )
                    mb_compounds.append(polymer)
                    self.system_parms.molecule_sequences.append(mol_sequence)
                    self.system_parms.para += mol_sequence.count("P")
                    self.system_parms.meta += mol_sequence.count("M")
            mass = n * sum(
                [ele.element_from_symbol(p.name).mass
                for p in polymer.particles()]
            )
            self.system_mass += mass  # amu
        return mb_compounds

    def _apply_ff(self, untyped_system):
        """Use foyer to type the system and store forcefield parameters.
        Returns a Parmed structure object.
        """
        ff_path = f"{FF_DIR}/{self.forcefield}.xml"
        forcefield = foyer.Forcefield(forcefield_files=ff_path)
        print("Applying a focefield:")
        typed_system = forcefield.apply(
                untyped_system, verbose=True, assert_dihedral_params=True
        )
        # Add charges to parmed struc from mbuild comp
        if self.charges:
            net_charge = sum([p.charge for p in untyped_system.particles()])
            abs_net = sum([abs(p.charge) for p in untyped_system.particles()])
            n_particles = untyped_system.n_particles
            adjust = abs(net_charge) / n_particles
            print("-----------------------------------------------------------")
            print(f"Net charge of {net_charge} for {n_particles} particles")
            print(f"Adjusting each charge by +/- {adjust}")
            for p, a in zip(untyped_system.particles(), typed_system.atoms):
                charge = p.charge
                charge = charge - (abs(charge) * (net_charge/abs_net))
                assert np.sign(charge) == np.sign(p.charge)
                a.charge = charge
            net_charge = sum([a.charge for a in typed_system.atoms])
            print(f"Resulting net charge of {net_charge}")
            print("-----------------------------------------------------------")
            print()
        return typed_system

    def _load_parmed_structure(self, untyped_system):
        """Loads the parmed structure from file, if it exists.
        Otherwise, creates the parmed structure and saves it
        to file using pickle.

        """
        if self.pmd_pickle_path and os.path.exists(self.pmd_pickle_path):
            # load parmed from file
            f = open(self.pmd_pickle_path, "rb")
            self.system = pickle.load(f)
            f.close()
        elif not self.cg_compounds:
            # Apply a forcefield to an atomistic system 
            self.system = self._apply_ff(untyped_system)
            if self.pmd_pickle_path:
                f = open(self.pmd_pickle_path, "wb")
                pickle.dump(self.system, f)
                f.close()
        else: # Coarse-grain the system
            gmso_system = from_mbuild(untyped_system)
            gmso_system.identify_connections()
            parmed_system = to_parmed(gmso_system)
            for atom in parmed_system.atoms:
                atom.type = atom.name
            self.system = parmed_system
            if self.pmd_pickle_path:
                f = open(self.pmd_pickle_path, "wb")
                pickle.dump(self.system, f)
                f.close()

    def _remove_hydrogens(self):
        # Adjust mass and charge of heavy atoms:
        print("Removing hydrogens and adjusting heavy atoms")
        init_net_charge = self.net_charge
        hydrogens = [a for a in self.system.atoms if a.element == 1]
        for h in hydrogens:
            bonded_atom = h.bond_partners[0]
            bonded_atom.mass += h.mass
            bonded_atom.charge += h.charge

        self.system.strip(
                [a.atomic_number == 1 for a in self.system.atoms]
        )
        assert np.allclose(init_net_charge, self.net_charge, atol=1e-4)


class Fused:
    def __init__(
            self, gsd_file, ref_distance, ref_energy, ref_mass, forcefield
    ):
        self.gsd_file = gsd_file
        self.ref_distance = ref_distance
        self.ref_energy = ref_energy
        self.ref_mass = ref_mass
        self.forcefield = forcefield
        self.system_type = "interface"

        system_mb = _gsd_to_mbuild(
                gsd_file=self.gsd_file,
                ref_distance=self.ref_distance,
                ref_energy=self.ref_energy,
                ref_mass=self.ref_mass
        )
        system_mb.box = mb.box.Box.from_mins_maxs_angles(
                mins=(0,0,0),
                maxs=system_mb.get_boundingbox().lengths,
                angles = (90, 90, 90)
        )
        system_mb.translate_to(
                [system_mb.box.Lx / 2,
                system_mb.box.Ly / 2,
                system_mb.box.Lz / 2,]
        )

        ff_path = f"{FF_DIR}/{self.forcefield}-nosmarts.xml"
        forcefield = foyer.Forcefield(forcefield_files=ff_path)
        self.system = forcefield.apply(system_mb)
        for p, a in zip(system_mb.particles(), self.system.atoms):
            a.charge = p.charge
            a.mass = p.mass
        net_charge = sum([a.charge for a in self.system.atoms])
        n_particles = system_mb.n_particles
        print("-----------------------------------------------------------")
        print(f"Net charge of {net_charge} for {n_particles} particles")


class Interface:
    """Initialize an interface system between one or two "slabs" that were
    created previously. Initializing a system via this class requires that the
    previous systems were ran using wall potentials to create two flat surfaces.

    Parameters
    ----------
    slabs : str, or list of str
        Path to the .gsd files to be used in creating an interface
    ref_distance : float
        The reference distance used to convert from simulation
        distance units to mBuild units (nm)
    ref_energy : float
        The reference energy used to convert from simulation energy units
        to the energy values used in the foyer forcefield (kJ/mol).
        Required for correctly setting the charge values
    forcefield : str
        Name of the focefield file used in the slab-generating simulations
    gap : float
        The size of the gap between interfaces (nm).
    weld_axis : str
        Set the axis to translate the slabs along.
        This should be the same axis used when setting the wall potentials
        for the single slab simulations.
        "x", "y" or "z"

    """
    def __init__(
            self,
            slabs,
            ref_distance,
            ref_energy,
            ref_mass,
            forcefield,
            gap=0.1,
            weld_axis="x"
    ):
        self.system_type = "interface"
        self.ref_distance = ref_distance
        self.ref_energy = ref_energy
        self.ref_mass = ref_mass
        self.forcefield = forcefield

        if not isinstance(slabs, list):
            slabs = [slabs]
        if len(slabs) == 2:
            slab_files = slabs
        else:
            slab_files = slabs * 2

        axis_dict = {
                "x": np.array([1,0,0]),
                "y": np.array([0,1,0]),
                "z": np.array([0,0,1])
        }
        weld_axis = weld_axis.lower()
        # Axis to translate slab systems along to create interface
        trans_axis = axis_dict[weld_axis]

        interface = mb.Compound()
        slab_1 = _gsd_to_mbuild(
                gsd_file=slab_files[0],
                ref_distance=self.ref_distance,
                ref_energy=self.ref_energy,
                ref_mass=self.ref_mass
        )
        slab_2 = _gsd_to_mbuild(
                gsd_file=slab_files[1],
                ref_distance=self.ref_distance,
                ref_energy=self.ref_energy,
                ref_mass=self.ref_mass
        )
        interface.add(new_child=slab_1, label="left")
        interface.add(new_child=slab_2, label="right")
        _len = getattr(interface.get_boundingbox(), f"L{weld_axis}")
        interface["left"].translate(-trans_axis*(_len+gap))
        system_box = mb.box.Box.from_mins_maxs_angles(
                mins=(0, 0, 0),
                maxs = interface.get_boundingbox().lengths,
                angles = (90, 90, 90)
        )
        current_len = getattr(system_box, f"L{weld_axis}")
        # Adjust interface box; account for LJ walls on ends of the weld axis
        setattr(system_box,
                f"_L{weld_axis}",
                current_len + (2*self.ref_distance*1.1225)
        )
        interface.box = system_box
        # Center in the adjusted box
        interface.translate_to(
                [interface.box.Lx / 2,
                interface.box.Ly / 2,
                interface.box.Lz / 2,]
        )

        ff_path = f"{FF_DIR}/{self.forcefield}-nosmarts.xml"
        forcefield = foyer.Forcefield(forcefield_files=ff_path)
        self.system = forcefield.apply(interface)
        for p, a in zip(interface.particles(), self.system.atoms):
            a.charge = p.charge
            a.mass = p.mass
        net_charge = sum([a.charge for a in self.system.atoms])
        n_particles = interface.n_particles
        print("-----------------------------------------------------------")
        print(f"Net charge of {net_charge} for {n_particles} particles")


def _gsd_to_mbuild(gsd_file, ref_distance, ref_energy, ref_mass):
    """Creates an mbuild.Compound system from a hoomd GSD file.
    Assumes that the positions and charges stored in the gsd file
    are the reduced values used during the simulation.

    """
    element_mapping = {
        "oh": "O",
        "ca": "C",
        "os": "O",
        "o": "O",
        "c": "C",
        "ho": "H",
        "ha": "H",
        "hs": "H",
        "s": "S",
        "sh": "S",

    }
    snap = trajectory = gsd.hoomd.open(gsd_file)[-1]
    e0 = 2.396452e-04
    charge_factor = (4.0 * np.pi * e0 * ref_distance * ref_energy) ** 0.5
    pos_wrap = snap.particles.position * ref_distance
    charges = snap.particles.charge * charge_factor
    masses = snap.particles.mass * ref_mass
    atom_types = [snap.particles.types[i] for i in snap.particles.typeid]
    elements = [element_mapping[i] for i in atom_types]

    comp = mb.Compound()
    for pos, charge, mass, element, atom_type in zip(
            pos_wrap, charges, masses, elements, atom_types):
        child = mb.Compound(
                name=f"_{atom_type}", pos=pos, charge=charge, mass=mass, element=element
        )
        comp.add(child)

    particle_dict = {}
    for idx, particle in enumerate(comp.particles()):
        particle_dict[idx] = particle
    
    bonds = [(i, j) for (i, j) in snap.bonds.group]
    for (i, j) in bonds:
        atom1 = particle_dict[i]
        atom2 = particle_dict[j]
        comp.add_bond(particle_pair=[atom1, atom2])
    return comp


def build_molecule(
        molecule, length, sequence, para_weight, smiles=False, charges=None
):
    """`build_molecule` uses SMILES strings to build up a polymer from monomers.
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
    charges : str, default None
        The value to use in the moleucle's JSON file
        This applies the specified partial charges to each atom

    Notes
    -----
    Any combined values entered for length and sequence should be compatible.
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
    else: # Generate sequence that matches polymer length
        n = length // len(sequence)
        monomer_sequence = sequence * n
        monomer_sequence += sequence[:(length - len(monomer_sequence))]
        monomer_sequence = "".join(monomer_sequence)

    compound = Polymer()
    if smiles:
        raise ValueError("SMILES strings in the compound .JSON files not currently supported")
    else:
        try:
            para = mb.load(os.path.join(COMPOUND_DIR, mol_dict["para_file"]))
            if charges:
                charge_dict = mol_dict["para_charges"][charges]
                para_head_charge = charge_dict[str(para.n_particles - 2)]
                para_tail_charge = charge_dict[str(para.n_particles - 1)]
                for idx, p in enumerate(para.particles()):
                    p.charge = charge_dict[str(idx)]
            if "M" in monomer_sequence:
                meta = mb.load(os.path.join(COMPOUND_DIR, mol_dict["meta_file"]))
                if charges:
                    charge_dict = mol_dict["meta_charges"][charges]
                    meta_head_charge = charge_dict[str(para.n_particles - 2)]
                    meta_tail_charge = charge_dict[str(para.n_particles - 1)]
                    for idx, p in enumerate(meta.particles()):
                        p.charge = charge_dict[str(idx)]
        except KeyError:
            print("No file is available for this compound")

    if len(set(monomer_sequence)) == 2: # Copolymer
        compound.add_monomer(
                meta,
                mol_dict["meta_bond_indices"],
                mol_dict["bond_distance"],
                mol_dict["bond_orientation"],
                replace=True
        )
        compound.add_monomer(
                para,
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
    if charges:
        if monomer_sequence[0] == "P":
            compound[-2].charge = para_head_charge
        elif monomer_sequence[0] == "M":
            compound[-2].charge = meta_head_charge
        if monomer_sequence[-1] == "P":
            compound[-1].charge = para_tail_charge
        elif monomer_sequence[-1] == "M":
            compound[-1].charge = meta_tail_charge
    # Rotate to align with z-axis; important for initializing ordered systems
    # Rotations are hard-coded based on the mol2 files in the library
    if molecule == "PEKK":
        compound.rotate(around=[0,1,0], theta=-0.20)
        compound.rotate(around=[1,0,0], theta=0.15)
    if molecule == "PPS":
        compound.rotate(around=[0,1,0], theta=0.05)
        compound.rotate(around=[1,0,0], theta=-0.010)
    return compound, monomer_sequence


def random_sequence(para_weight, length):
    """Creates a list containing a random sequence of strings
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
