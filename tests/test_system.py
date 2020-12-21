import pytest

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
    def test_load_forcefiled(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[1], polymer_lengths=[2],
                               forcefield=None, assert_dihedrals=False, remove_hydrogens=True)

    # test our dihedral checking
    def test_load_forcefiled(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[1], polymer_lengths=[2],
                               forcefield=None, assert_dihedrals=True, remove_hydrogens=False)

    # make sure we can handle several compounds of several lengths
    def test_load_forcefiled(self):
        simple_system = System(molecule='PEEK', para_weight=0.60,
                               density=1, n_compounds=[5, 4, 3, 2, 1], polymer_lengths=[2, 4, 5, 11, 22],
                               forcefield='gaff', assert_dihedrals=False, remove_hydrogens=False)