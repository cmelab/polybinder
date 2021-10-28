import os
import pytest
import random

import numpy as np
import uli_init
from uli_init.system import System, Initializer
from uli_init.library import SYSTEM_DIR
from base_test import BaseTest


class TestSystems(BaseTest):
    def test_set_box(self):
        system_params = System(
                    density=1.0,
                    molecule="PEKK",
                    para_weight=1.0,
                    n_compounds=10,
                    polymer_lengths=5
                )
        system = Initializer(
                system_params, system_type="pack"
            )
        init_box = system.target_box
        system.set_target_box(x_constraint = init_box[0]/2)
        new_box = system.target_box

        assert np.allclose(
                np.prod(init_box), np.prod(new_box), atol=1e-3
            )
        assert np.allclose(new_box[1], new_box[2], atol=1e-3)
        assert np.allclose(
                new_box[1]**2,  2*(init_box[0]**2), atol=1e-3
            )

    def test_bad_system_type(self):
        with pytest.raises(ValueError):
            system_params = System(
                        density=0.7,
                        molecule="PEEK",
                        para_weight=0.50,
                        n_compounds=10,
                        polymer_lengths=5,
                    )
            system = Initializer(
                    system_params, system_type="wrong"
                    )

    def test_pack(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=0.50,
                density=0.7,
                n_compounds=[10],
                polymer_lengths=[5],
                )
        system = Initializer(
                system_parms, system_type="pack", expand_factor=10
                )


    def test_stack(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=0.50,
                density=0.7,
                n_compounds=[10],
                polymer_lengths=[5],
                )
        system = Initializer(
                system_parms, system_type="stack", separation=1.0
                )

    def test_crystal(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=1.0,
                density=0.7,
                n_compounds=[8],
                polymer_lengths=[5],
                )
        system = Initializer(
                system_parms, system_type="crystal",
                a = 1.5, b = 0.9, n=2
                )

    def test_crystal_bad_n(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=1.0,
                density=0.7,
                n_compounds=[12],
                polymer_lengths=[5],
                )
        with pytest.raises(ValueError):
            system = Initializer(
                system_parms, system_type="crystal",
                a = 1.5, b = 0.9, n=2
                )


    def test_build_peek(self):
        for i in range(5):
            compound = uli_init.system.build_molecule(
                    "PEEK", i + 1,
                    sequence = "random",
                    para_weight=0.50
                    )

    def test_build_pekk(self):
        for i in range(5):
            compound = uli_init.system.build_molecule(
                    "PEKK", 
                    i + 1,
                    sequence="random",
                    para_weight=0.50
                    )

    def test_monomer_sequence(self):
        with pytest.warns(UserWarning):
            system_parms = System(
                    molecule="PEEK",
                    monomer_sequence="PM",
                    para_weight=0.5,
                    n_compounds = [1],
                    polymer_lengths=[4],
                    density=.1,
                    )

        system_parms_even = System(
                molecule="PEEK",
                monomer_sequence="PM",
                n_compounds = [1],
                polymer_lengths=[4],
                density=.1,
                )
        system_even = Initializer(system_parms_even, "pack")

        assert system_parms_even.para == system_parms_even.meta  == 2
        assert system_parms_even.molecule_sequences[0] == "PMPM"

        system_parms = System(
                molecule="PEEK",
                monomer_sequence="PM",
                n_compounds = [1],
                polymer_lengths=[5],
                density=.1,
                )
        system_odd = Initializer(
                system_parms, "pack"
                )
        assert system_parms.para == 3
        assert system_parms.meta == 2
        assert system_parms.molecule_sequences[0] == "PMPMP"

        system_parms = System(
                molecule="PEEK",
                monomer_sequence="PMPMPMPMPM",
                n_compounds = [1],
                polymer_lengths=[4],
                density=.1,
                )
        system_long_seq = Initializer(
                system_parms, "pack"
                )
        assert system_parms.para == system_parms.meta == 2
        assert system_parms.molecule_sequences[0] == "PMPM"

    def test_para_weight(self):
        all_para = uli_init.system.random_sequence(para_weight=1, length=10)
        all_meta = uli_init.system.random_sequence(para_weight=0, length=10)
        assert all_para.count("P") == 10
        assert all_meta.count("M") == 10
        random.seed(24)
        mix_sequence = uli_init.system.random_sequence(para_weight=0.50, length=10)
        assert mix_sequence.count("P") == 4
        random.seed()
        mix_sequence = uli_init.system.random_sequence(para_weight=0.70, length=100)
        assert 100 - mix_sequence.count("P") == mix_sequence.count("M")
    
    def test_weighted_sequence(self):
        para_monomers = System(
                molecule="PEEK",
                para_weight = 1.0,
                n_compounds = [1],
                polymer_lengths=[10],
                density=0.1,
                )
        para_system = Initializer(para_monomers, "pack")
        assert para_monomers.molecule_sequences[0] == "P"*10

        random_monomers = System(
                molecule="PEEK",
                para_weight = 0.40,
                n_compounds = [1],
                polymer_lengths=[20],
                density=0.1,
                )
        random_system = Initializer(random_monomers, "pack", expand_factor=10)
        random.seed(24)
        sequence = uli_init.system.random_sequence(para_weight=0.40, length=20)
        sequence = "".join(sequence)
        assert random_monomers.molecule_sequences[0] == sequence

    def test_load_forcefiled(self):
        system_parms = System(
            molecule="PEEK",
            para_weight=0.60,
            density=.1,
            n_compounds=[1],
            polymer_lengths=[2],
        )
        simple_system = Initializer(
                system_parms, "pack", forcefield="gaff"
                )

    def test_remove_hydrogens(self):
        system_parms = System(
            molecule="PEEK",
            para_weight=0.60,
            density=.1,
            n_compounds=[1],
            polymer_lengths=[2],
        )
        system = Initializer(
                system_parms, "pack", forcefield="gaff", remove_hydrogens=True
                )
        post_remove_h = system.system
        assert sum([int(item.type == "H") for item in post_remove_h.atoms]) == 0

    def test_multiple_compounds(self):
        system_parms = System(
            molecule="PEEK",
            para_weight=0.60,
            density=.1,
            n_compounds=[5, 4, 3, 2, 1],
            polymer_lengths=[2, 4, 5, 11, 22],
        )
        system = Initializer(
                system_parms, "pack"
                )

    def test_pdi_mw(self):
        system_parms = System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mw=6.0,
        )
        system = Initializer(system_parms, "pack")


    def test_pdi_mn(self):
        system_parms = System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mn=5.0,
        )
        system = Initializer(system_parms, "pack")


    def test_mw_mn(self):
        system_parms = System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            Mn=5.0,
            Mw=6.0,
        )
        system = Initializer(system_parms, "pack")


    def test_pdi_mn_mw(self):
        system_parms = System(
            molecule="PEEK",
            para_weight=0.60,
            density=0.1,
            n_compounds=3,
            sample_pdi=True,
            pdi=1.2,
            Mw=6.0,
            Mn=5.0,
        )
        system = Initializer(system_parms, "pack")


    def test_too_few_pdi_vals(self):
        with pytest.raises(AssertionError):
            system_parms = System(
                molecule="PEEK",
                para_weight=0.60,
                density=0.1,
                n_compounds=3,
                sample_pdi=True,
                pdi=1.2,
            )


    def test_incorrect_pdi_vals(self):
        with pytest.raises(AssertionError):
            system_parms = System(
                molecule="PEEK",
                para_weight=0.60,
                density=0.1,
                n_compounds=3,
                sample_pdi=True,
                pdi=1.2,
                Mn=5.0,
                Mw=8.0,
            )
