import os
import pytest
import random

from uli_init import simulate, system
from uli_init.library import SYSTEM_DIR
from base_test import BaseTest


class TestSystems(BaseTest):
    def test_custom_bad_params(self):
        file_path = os.path.join(
                SYSTEM_DIR,
                "test_peek.mol2"
                )
        with pytest.warns(UserWarning):
            custom_sys = system.System(
                    system_type = "custom",
                    density=0.7,
                    molecule="PEEK",
                    para_weight=0.50,
                    n_compounds=10,
                    polymer_lengths=5,
                    file_path=file_path
                    )
            ignore_args = [
                    custom_sys.molecule,
                    custom_sys.para_weight,
                    custom_sys.polymer_lengths,
                    custom_sys.n_compounds
                    ]
            assert not any(ignore_args)

    def test_stack(self):
        stacked_system = system.System(
                molecule="PEEK",
                para_weight=0.50,
                density=0.7,
                n_compounds=[10],
                polymer_lengths=[5],
                system_type="stack",
                forcefield="gaff",
                remove_hydrogens=False,
                expand_factor=4
                )

    def test_build_peek(self):
        for i in range(5):
            compound = system.build_molecule(
                    "PEEK", i + 1,
                    sequence = "random",
                    para_weight=0.50
                    )

    def test_build_pekk(self):
        for i in range(5):
            compound = system.build_molecule(
                    "PEKK", 
                    i + 1,
                    sequence="random",
                    para_weight=0.50
                    )

    def test_monomer_sequence(self):
        with pytest.warns(UserWarning):
            system_even = system.System(
                    molecule="PEEK",
                    monomer_sequence="PM",
                    para_weight=0.5,
                    n_compounds = [1],
                    polymer_lengths=[4],
                    density=.1,
                    system_type="pack",
                    remove_hydrogens=True
                    )

        system_even = system.System(
                molecule="PEEK",
                monomer_sequence="PM",
                n_compounds = [1],
                polymer_lengths=[4],
                density=.1,
                system_type="pack",
                remove_hydrogens=True
                )
        assert system_even.para == system_even.meta  == 2
        assert system_even.molecule_sequences == ["PMPM"]

        system_odd = system.System(
                molecule="PEEK",
                monomer_sequence="PM",
                n_compounds = [1],
                polymer_lengths=[5],
                density=.1,
                system_type="pack",
                remove_hydrogens=True
                )
        assert system_odd.para == 3
        assert system_odd.meta == 2
        assert system_odd.molecule_sequences == ["PMPMP"]

        system_large_seq = system.System(
                molecule="PEEK",
                monomer_sequence="PMPMPMPMPM",
                n_compounds = [1],
                polymer_lengths=[4],
                density=.1,
                system_type="pack",
                remove_hydrogens=True
                )
        assert system_large_seq.para == system_large_seq.meta == 2

    def test_para_weight(self):
        all_para = system.random_sequence(para_weight=1, length=10)
        all_meta = system.random_sequence(para_weight=0, length=10)
        assert all_para.count("P") == 10
        assert all_meta.count("M") == 10
        random.seed(24)
        mix_sequence = system.random_sequence(para_weight=0.50, length=10)
        assert mix_sequence.count("P") == 4
        random.seed()
        mix_sequence = system.random_sequence(para_weight=0.70, length=100)
        assert 100 - mix_sequence.count("P") == mix_sequence.count("M")
    
    def test_weighted_sequence(self):
        para_system = system.System(
                molecule="PEEK",
                para_weight = 1.0,
                n_compounds = [1],
                polymer_lengths=[10],
                density=0.1,
                system_type="pack"
                )
        assert para_system.molecule_sequences[0] == ["P"]*10

        random_system = system.System(
                molecule="PEEK",
                para_weight = 0.40,
                n_compounds = [1],
                polymer_lengths=[20],
                density=0.1,
                system_type="pack"
                )
        random.seed(24)
        sequence = system.random_sequence(para_weight=0.40, length=20)
        assert random_system.molecule_sequences[0] == sequence

    def test_load_forcefiled(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=.1,
            n_compounds=[1],
            polymer_lengths=[2],
            system_type="pack",
            forcefield="gaff",
            assert_dihedrals=False,
            remove_hydrogens=False,
        )

    def test_remove_hydrogens(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=.1,
            n_compounds=[1],
            polymer_lengths=[2],
            system_type="pack",
            forcefield="gaff",
            assert_dihedrals=False,
            remove_hydrogens=True,
        )
        post_remove_h = simple_system.system
        assert sum([int(item.type == "H") for item in post_remove_h.atoms]) == 0

    def test_dihedrals(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=.1,
            n_compounds=[1],
            polymer_lengths=[2],
            forcefield="gaff",
            system_type="pack",
            assert_dihedrals=True,
            remove_hydrogens=False,
        )

    def test_multiple_compounds(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=.1,
            n_compounds=[5, 4, 3, 2, 1],
            polymer_lengths=[2, 4, 5, 11, 22],
            forcefield=None,
            system_type="pack",
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_pdi_mw(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mw=6.0,
            system_type="pack",
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_pdi_mn(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mn=5.0,
            system_type="pack",
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_mw_mn(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            Mn=5.0,
            Mw=6.0,
            system_type="pack",
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_pdi_mn_mw(self):
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            system_type="pack",
            pdi=1.2,
            Mw=6.0,
            Mn=5.0,
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            expand_factor = 7
        )

    def test_too_few_pdi_vals(self):
        with pytest.raises(AssertionError):
            simple_system = system.System(
                molecule="PEEK",
                para_weight=0.60,
                density=0.1,
                n_compounds=3,
                sample_pdi=True,
                pdi=1.2,
                forcefield=None,
                system_type="pack",
                assert_dihedrals=False,
                remove_hydrogens=False,
            )

    def test_incorrect_pdi_vals(self):
        with pytest.raises(AssertionError):
            simple_system = system.System(
                molecule="PEEK",
                para_weight=0.60,
                density=0.1,
                n_compounds=3,
                sample_pdi=True,
                pdi=1.2,
                Mn=5.0,
                Mw=8.0,
                system_type="pack",
                forcefield=None,
                assert_dihedrals=False,
                remove_hydrogens=False,
            )

    def test_gauss_dist(self):
        random.seed(42)
        simple_system = system.System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.001,
            Mn=5.0,
            system_type="pack",
            forcefield=None,
            assert_dihedrals=False,
            remove_hydrogens=False,
            mass_dist_type="gaussian",
        )
