import os

import pytest
from uli_init import simulate, system
from uli_init.library import COMPOUND_DIR, SYSTEM_DIR

class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture()
    def peek_from_file(self):
        peek = os.path.join(
                SYSTEM_DIR,
                "test_peek.mol2"
                )
        system_parms = system.System(
                    density=0.8,
                    forcefield="gaff",
                    file_path=peek
                    )
        system = system.Initialize(
                system_parms,
                "custom",
                remove_hydrogens=True
                )

        return system

    @pytest.fixture()
    def test_slab(self):
        slab = system.Interface(
            slabs=os.path.join(
                SYSTEM_DIR,
                "test-slab.gsd"),
            ref_distance=0.33996695084235347,
            gap=0.1
        )
        return slab

    @pytest.fixture
    def peek_system(self):
        system_parms = system.System(
                molecule="PEEK",
                para_weight=0.50,
                density=1.2,
                n_compounds=[3],
                polymer_lengths=[3],
        )
        peek_sys = system.Initialize(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=False
                )
        return peek_sys

    @pytest.fixture
    def pekk_system(self):
        system_parms = system.System(
                molecule="PEKK",
                para_weight=0.50,
                density=1.2,
                n_compounds=[3],
                polymer_lengths=[3],
        )
        pekk_sys = system.Initialize(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=False
                )
        return pekk_sys

    @pytest.fixture
    def peek_system_noH(self):
        system_parms = system.System(
                molecule="PEEK",
                para_weight=0.50,
                density=1.2,
                n_compounds=[3],
                polymer_lengths=[3],
        )
        peek_sys = system.Initialize(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=True
                )
        return peek_sys

    @pytest.fixture
    def pekk_system_noH(self):
        system_parms = system.System(
                molecule="PEKK",
                para_weight=0.50,
                density=1.2,
                n_compounds=[3],
                polymer_lengths=[3],
        )
        pekk_sys = system.Initialize(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=True
                )
        return pekk_sys
        return pekk_sys
