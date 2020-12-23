import pytest
from numpy import random
from base_test import BaseTest
from uli_init.simulate import System

class TestSystems(BaseTest):
    # check the base functionality
    def test_simple_system(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[1], polymer_lengths=[2],
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    # check that we can load forcefield files
    def test_load_forcefiled(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[1], polymer_lengths=[2],
                               forcefield='gaff', assert_dihedrals=False, remove_hydrogens=False)

    # check that we can remove H atoms
    def test_remove_hydrogens(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[1], polymer_lengths=[2],
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=True)

    # test our dihedral checking
    def test_test_dihedrals(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[1], polymer_lengths=[2],
                               forcefield=None, assert_dihedrals=True, remove_hydrogens=False)

    # make sure we can handle several compounds of several lengths
    def test_multiple_compounds(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[5, 4, 3, 2, 1], polymer_lengths=[2, 4, 5, 11, 22],
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    # test PDI sampling with four combinations of argument values
    def test_pdi_mw(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=3, sample_pdi=True, pdi=1.2, Mw=6.,
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    def test_pdi_mn(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=3, sample_pdi=True, pdi=1.2, Mn=5.,
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    def test_mw_mn(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=3, sample_pdi=True, Mn=5., Mw=6.,
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    def test_pdi_mn_mw(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=3, sample_pdi=True, pdi=1.2, Mw=6., Mn=5.,
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    # make sure we correctly throw error when not enough PDI args
    def test_too_few_pdi_vals(self):
        with pytest.raises(AssertionError):
            simple_system = System(molecule='PEEK', para_weight=0.60,
                                density=1, n_compounds=3, sample_pdi=True, pdi=1.2,
                                forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    # make sure we correctly throw error when PDI args don't match up
    def test_incorrect_pdi_vals(self):
        with pytest.raises(AssertionError):
            simple_system = System(molecule='PEEK', para_weight=0.60,
                                density=1, n_compounds=3, sample_pdi=True, pdi=1.2, Mn=5., Mw=8.,
                                forcefield=None, assert_dihedrals=False, remove_hydrogens=False)

    # weibull is the default tpye, so test gaussian also 
    def test_gauss_dist(self):
        random.seed(42)
        simple_system = System(molecule='PEEK', para_weight=0.60,
                            density=1, n_compounds=3, sample_pdi=True, pdi=1.001, Mn=50.,
                            forcefield=None, assert_dihedrals=False, remove_hydrogens=False,
                            mass_dist_type='gaussian')

    
