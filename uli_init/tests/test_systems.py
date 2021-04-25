import pytest
import random

from uli_init import simulate
from uli_init.tests.base_test import BaseTest


class TestSystems(BaseTest):

    def test_single_chain(self):
        chain = simulate.System(
                molecule="PEEK",
                para_weight=1.0,
                density=0.10,
                n_compounds=1,
                polymer_lengths=10,
                forcefield="gaff",
                remove_hydrogens=True
                )

    def test_build_peek(self):
        for i in range(5):
            compound = simulate.build_molecule("PEEK", i + 1, para_weight=0.50)

    def test_build_pekk(self):
        for i in range(5):
            compound = simulate.build_molecule("PEKK", i + 1, para_weight=0.50)

    def test_para_weight(self):
        all_para = simulate.random_sequence(para_weight=1, length=10)
        all_meta = simulate.random_sequence(para_weight=0, length=10)
        assert all_para.count("P") == 10
        assert all_meta.count("M") == 10
        random.seed(24)
        mix_sequence = simulate.random_sequence(para_weight=0.50, length=10)
        assert mix_sequence.count("P") == 4
        random.seed()
        mix_sequence = simulate.random_sequence(para_weight=0.70, length=100)
        assert 100 - mix_sequence.count("P") == mix_sequence.count("M")

    # check that we can load forcefield files
    def test_load_forcefiled(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=1,
            n_compounds=[1],
            polymer_lengths=[2],
            forcefield="gaff",
            assert_dihedrals=False,
            remove_hydrogens=False,
        )

    # check that we can remove H atoms
    def test_remove_hydrogens(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=1,
            n_compounds=[1],
            polymer_lengths=[2],
            forcefield="gaff",
            assert_dihedrals=False,
            remove_hydrogens=True,
        )
        # mbuild Compound object (does not have H removed)
        pre_remove_h = simple_system.system_mb
        # parmed AtomList object after removing H
        post_remove_h = simple_system.system_pmd
        # make sure the atom count changed
        assert pre_remove_h.n_particles > len(post_remove_h.atoms)
        # make sure there are actually zero H atoms
        assert sum([int(item.type == "H") for item in post_remove_h.atoms]) == 0

    # test our dihedral checking
    def test_dihedrals(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=1,
            n_compounds=[1],
            polymer_lengths=[2],
            forcefield="gaff",
            assert_dihedrals=True,
            remove_hydrogens=False,
        )

    # make sure we can handle several compounds of several lengths
    def test_multiple_compounds(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=1,
            n_compounds=[5, 4, 3, 2, 1],
            polymer_lengths=[2, 4, 5, 11, 22],
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    # test PDI sampling with four combinations of argument values
    def test_pdi_mw(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.7,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mw=6.0,
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_pdi_mn(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.7,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mn=5.0,
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_mw_mn(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.7,
            n_compounds=3,
            sample_pdi=True,
            Mn=5.0,
            Mw=6.0,
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_pdi_mn_mw(self):
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.7,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mw=6.0,
            Mn=5.0,
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    # make sure we correctly throw error when not enough PDI args
    def test_too_few_pdi_vals(self):
        with pytest.raises(AssertionError):
            simple_system = simulate.System(
                molecule="PEEK",
                para_weight=0.60,
                density=0.7,
                n_compounds=3,
                sample_pdi=True,
                pdi=1.2,
                forcefield=None,
                assert_dihedrals=False,
                remove_hydrogens=False,
            )

    # make sure we correctly throw error when PDI args don't match up
    def test_incorrect_pdi_vals(self):
        with pytest.raises(AssertionError):
            simple_system = simulate.System(
                molecule="PEEK",
                para_weight=0.60,
                density=0.7,
                n_compounds=3,
                sample_pdi=True,
                pdi=1.2,
                Mn=5.0,
                Mw=8.0,
                forcefield=None,
                assert_dihedrals=False,
                remove_hydrogens=False,
            )

    # weibull is the default tpye, so test gaussian also
    def test_gauss_dist(self):
        random.seed(42)
        simple_system = simulate.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.7,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.001,
            Mn=5.0,
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            mass_dist_type="gaussian",
        )
