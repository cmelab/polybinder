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
            ref_energy=0.21,
            ref_mass=15.98,
            gap=0.1,
            forcefield="gaff",
            weld_axis="x"
        )
        return slab

    @pytest.fixture()
    def test_interface_y(self):
        slab = Interface(
            slabs=os.path.join(ASSETS_DIR, "test_slab_ywall.gsd"),
            ref_distance=0.33997,
            ref_energy=0.21,
            ref_mass=15.98,
            gap=0.1,
            forcefield="gaff",
            weld_axis="y"
        )
        return slab

    @pytest.fixture()
    def test_interface_z(self):
        slab = Interface(
            slabs=os.path.join(ASSETS_DIR, "test_slab_zwall.gsd"),
            ref_distance=0.33997,
            ref_energy=0.21,
            ref_mass=15.98,
            gap=0.1,
            weld_axis="z",
            forcefield="gaff",
        )
        return slab

    @pytest.fixture
    def cg_system_components(self):
        system_parms = System(
                molecule="PEKK",
                para_weight=1.0,
                density=0.08,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        pekk_sys = Initializer(system_parms, forcefield=None, save_parmed=False)
        pekk_sys.coarse_grain_system(
                use_components=True,
                bead_mapping="ring_plus_linkage_AA"
        )
        pekk_sys.pack()
        return pekk_sys

    @pytest.fixture
    def cg_system_monomers(self):
        system_parms = System(
                molecule="PEKK",
                para_weight=1.0,
                density=0.08,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        pekk_sys = Initializer(system_parms, forcefield=None, save_parmed=False)
        pekk_sys.coarse_grain_system(use_monomers=True)
        pekk_sys.pack()
        return pekk_sys

    @pytest.fixture
    def pps_system(self):
        system_parms = System(
                molecule="PPS",
                para_weight=1.0,
                density=0.8,
                n_compounds=[10],
                polymer_lengths=[2],
        )
        pps_sys = Initializer(
                system_parms, forcefield="pps_opls", remove_hydrogens=False
        )
        pps_sys.pack()
        return pps_sys

    @pytest.fixture
    def pps_system_charges(self):
        system_parms = System(
                density=1.20,
                molecule="PPS",
                para_weight=1.0,
                polymer_lengths=1,
                n_compounds=5
        )
        pps_sys = Initializer(
            system_parms,
            forcefield="pps_opls",
            charges="antechamber",
            remove_hydrogens=False
        )
        pps_sys.pack()
        return pps_sys

    @pytest.fixture
    def peek_system(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=1.0,
                density=0.8,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        peek_sys = Initializer(
                system_parms, forcefield="gaff", remove_hydrogens=False
        )
        peek_sys.pack()
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
                system_parms, forcefield="gaff", remove_hydrogens=False
        )
        pekk_sys.pack()
        return pekk_sys

    @pytest.fixture
    def peek_system_noH(self):
        system_parms = System(
                molecule="PEEK",
                para_weight=1.0,
                density=0.5,
                n_compounds=[10],
                polymer_lengths=[3],
        )
        peek_sys = Initializer(
                system_parms, forcefield="gaff", remove_hydrogens=True
        )
        peek_sys.pack()
        return peek_sys

    @pytest.fixture
    def pps_system_noH(self):
        system_parms = System(
                molecule="PPS",
                para_weight=1.0,
                density=0.5,
                n_compounds=[10],
                polymer_lengths=[2],
        )
        pps_sys = Initializer(
                system_parms, forcefield="pps_opls", remove_hydrogens=True
        )
        pps_sys.pack()
        return pps_sys

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
                system_parms, forcefield="gaff", remove_hydrogens=True
        )
        pekk_sys.pack()
        return pekk_sys

    @pytest.fixture
    def restart_gsd(self):
        return os.path.join(ASSETS_DIR, "test_restart.gsd") 
