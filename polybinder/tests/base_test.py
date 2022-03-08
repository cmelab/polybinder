import os

import pytest
from polybinder.system import System, Initializer, Interface
from polybinder.library import COMPOUND_DIR, SYSTEM_DIR, ASSETS_DIR


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture()
    def test_interface_x(self):
        slab = Interface(
            slabs=os.path.join(ASSETS_DIR, "test_slab_xwall.gsd"),
            ref_distance=0.33997,
            gap=0.1,
            weld_axis="x"
        )
        return slab

    @pytest.fixture()
    def test_interface_y(self):
        slab = Interface(
            slabs=os.path.join(ASSETS_DIR, "test_slab_ywall.gsd"),
            ref_distance=0.33997,
            gap=0.1,
            weld_axis="y"
        )
        return slab

    @pytest.fixture()
    def test_interface_z(self):
        slab = Interface(
            slabs=os.path.join(ASSETS_DIR, "test_slab_zwall.gsd"),
            ref_distance=0.33997,
            gap=0.1,
            weld_axis="z"
        )
        return slab

    @pytest.fixture
    def cg_system(self):
        system_parms = System(
                molecule="PEKK",
                para_weight=1.0,
                density=0.08,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        pekk_sys = Initializer(
                system_parms,
                system_type="pack",
                forcefield=None,
                remove_hydrogens=True
                )
        pekk_sys.coarse_grain_system(
                ref_distance=3.39, ref_mass=15.99
        )
        return pekk_sys

    @pytest.fixture
    def peek_system(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=0.50,
                density=0.8,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        peek_sys = Initializer(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=False
                )
        return peek_sys

    @pytest.fixture
    def pekk_system(self):
        system_parms = System(
                molecule="PEKK",
                para_weight=0.50,
                density=0.8,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        pekk_sys = Initializer(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=False
                )
        return pekk_sys

    @pytest.fixture
    def peek_system_noH(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=0.50,
                density=0.8,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        peek_sys = Initializer(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=True
                )
        return peek_sys

    @pytest.fixture
    def pekk_system_noH(self):
        system_parms = System(
                molecule="PEKK",
                para_weight=0.50,
                density=0.8,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        pekk_sys = Initializer(
                system_parms,
                system_type="pack",
                forcefield="gaff",
                remove_hydrogens=True
                )
        return pekk_sys
