from polybinder import simulate
from base_test import BaseTest

import pytest


class TestSimulate(BaseTest):

    def test_bad_inputs(self, peek_system, cg_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        # Only 1 of shrink params given
        with pytest.raises(ValueError):
            simulation.quench(
                kT=2,
                n_steps=5e2,
                shrink_kT=5,
                shrink_steps=None,
                shrink_period=None,
            )
        # Pressure and walls quench
        with pytest.raises(ValueError):
            simulation.quench(
                kT=2,
                pressure=0.1,
                n_steps=5e2,
                shrink_kT=None,
                shrink_steps=None,
                shrink_period=None,
                wall_axis=[1, 0, 0],
            )
        # Pressure and walls anneal
        with pytest.raises(ValueError):
            simulation.anneal(
                kT_init=2,
                kT_final=1,
                step_sequence = [5e2, 5e2],
                pressure=0.1,
                shrink_kT=None,
                shrink_steps=None,
                shrink_period=None,
                wall_axis=[1, 0 , 0],
            )
        with pytest.raises(AssertionError):
            sim = simulate.Simulation(
                    system=cg_system,
                    auto_scale=True,
                    ref_values=None
            )

    def test_quench_no_shrink(self, peek_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.quench(
            kT=2,
            pressure=0.1,
            n_steps=5e2,
            shrink_kT=None,
            shrink_steps=None,
            shrink_period=None,
        )

    def test_anneal_no_shrink(self, peek_system):
        simulation = simulate.Simulation(peek_system, dt=0.0001, mode="cpu")
        simulation.anneal(
            kT_init=4,
            kT_final=2,
            step_sequence=[5e2, 5e2],
            shrink_kT=None,
            shrink_steps=None,
            shrink_period=None,
        )

    def test_quench_npt(self, peek_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.quench(
            kT=2,
            pressure=0.1,
            n_steps=5e2,
            shrink_kT=8,
            shrink_steps=1e3,
            shrink_period=1,
        )

    def test_quench_nvt(self, peek_system):
        simulation = simulate.Simulation(peek_system, dt=0.0001, mode="cpu")
        simulation.quench(
            kT=2,
            n_steps=1e3,
            shrink_kT=8,
            shrink_steps=5e2,
            shrink_period=1,
        )

    def test_anneal_npt(self, pekk_system):
        simulation = simulate.Simulation(
                pekk_system,
                dt=0.0001,
                tau_p=0.1,
                mode="cpu"
        )
        simulation.anneal(
            kT_init=4,
            kT_final=2,
            pressure=0.1,
            step_sequence=[5e2, 5e2],
            shrink_kT=8,
            shrink_steps=5e2,
            shrink_period=1,
        )

    def test_anneal_nvt(self, pekk_system):
        simulation = simulate.Simulation(pekk_system, dt=0.0001, mode="cpu")
        simulation.anneal(
            kT_init=4,
            kT_final=2,
            step_sequence=[5e2, 5e2],
            shrink_kT=8,
            shrink_steps=5e2,
            shrink_period=1,
        )

    def test_quench_noH(self, pekk_system_noH):
        simulation = simulate.Simulation(pekk_system_noH, dt=0.001, mode="cpu")
        simulation.quench(
            kT=2,
            n_steps=5e2,
            shrink_kT=8,
            shrink_steps=5e2,
            shrink_period=1,
        )

    def test_anneal_noH(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt=0.001, mode="cpu")
        simulation.anneal(
            kT_init=4,
            kT_final=2,
            step_sequence=[5e2, 5e2],
            shrink_kT=8,
            shrink_steps=5e2,
            shrink_period=1,
        )

    def test_walls_x_quench(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt=0.001, mode="cpu")
        simulation.quench(
            kT=2,
            n_steps=1e2,
            shrink_kT=6,
            shrink_steps=1e3,
            shrink_period=5,
            wall_axis=[1,0,0],
        )

    def test_walls_x_anneal(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt=0.001, mode="cpu")
        simulation.anneal(
            kT_init=4,
            kT_final=2,
            step_sequence=[1e2, 1e2],
            shrink_kT=6,
            shrink_steps=1e3,
            shrink_period=5,
            wall_axis=[1,0,0],
        )

    def test_walls_y_quench(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt=0.001, mode="cpu")
        simulation.quench(
            kT=2,
            n_steps=1e2,
            shrink_kT=6,
            shrink_steps=1e3,
            shrink_period=5,
            wall_axis=[0,1,0],
        )

    def test_walls_y_anneal(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt=0.001, mode="cpu")
        simulation.anneal(
            kT_init=4,
            kT_final=2,
            step_sequence=[1e2, 1e2],
            shrink_kT=6,
            shrink_steps=1e3,
            shrink_period=5,
            wall_axis=[0,1,0],
        )

    def test_walls_z_quench(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt=0.001, mode="cpu")
        simulation.quench(
            kT=2,
            n_steps=1e2,
            shrink_kT=6,
            shrink_steps=1e3,
            shrink_period=5,
            wall_axis=[0,0,1],
        )

    def test_walls_z_anneal(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt=0.001, mode="cpu")
        simulation.anneal(
            kT_init=4,
            kT_final=2,
            step_sequence=[1e2, 1e2],
            shrink_kT=6,
            shrink_steps=1e3,
            shrink_period=5,
            wall_axis=[0,0,1],
        )

    def test_weld_quench(self, test_interface_x):
        simulation = simulate.Simulation(test_interface_x, dt=0.0001, mode="cpu")
        simulation.quench(kT=2, n_steps=5e2, wall_axis=[1,0,0])

    def test_weld_anneal(self, test_interface_y):
        simulation = simulate.Simulation(test_interface_y, dt=0.0001, mode="cpu")
        simulation.anneal(
            kT_init=4, kT_final=2, step_sequence=[5e2, 5e2, 5e2], wall_axis=[0,1,0]
        )

    def test_tensile_x(self, test_interface_x):
        simulation = simulate.Simulation(test_interface_x, dt=0.00001, mode="cpu")
        simulation.tensile(
                kT=2.0,
                fix_ratio = 0.30,
                strain=0.25,
                n_steps=5e2,
                expand_period=10
        )

    def test_tensile_y(self, test_interface_y):
        simulation = simulate.Simulation(test_interface_y, dt=0.00001, mode="cpu")
        simulation.tensile(
                kT=2.0,
                tensile_axis="y",
                fix_ratio = 0.30,
                strain=0.25,
                n_steps=1e3,
                expand_period=10
        )

    def test_tensile_z(self, test_interface_z):
        simulation = simulate.Simulation(test_interface_z, dt=0.00001, mode="cpu")
        simulation.tensile(
                kT=2.0,
                tensile_axis="z",
                fix_ratio = 0.30,
                strain=0.25,
                n_steps=1e3,
                expand_period=10
        )

    def test_cg_sim(self, cg_system):
        simulation = simulate.Simulation(
                cg_system,
                r_cut=2.5,
                mode="cpu",
                ref_values = {"distance": 3.3997, "energy": 0.21, "mass": 15.99},
                cg_potentials_dir = "test_potentials"
        )
        simulation.quench(
                kT=3.5,
                n_steps=500,
                shrink_kT=3.5,
                shrink_steps=500,
                shrink_period=10,
                table_pot=True
        )
