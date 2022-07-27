import hoomd
from polybinder import simulate
from base_test import BaseTest

import pytest


class TestSimulate(BaseTest):
    def test_charges_sim(self, pps_system_charges):
        simulation = simulate.Simulation(
                pps_system_charges,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.temp_ramp(
                kT_init=0.5, kT_final=1.5, n_steps=5e2, pressure=0.003
        )

    def test_temp_ramp_npt_walls(self, pps_system):
        with pytest.raises(ValueError):
            simulation = simulate.Simulation(
                    pps_system,
                    tau_p=0.1,
                    dt=0.0001,
                    mode="cpu",
                    wall_axis=[1,0,0]
            )
            simulation.temp_ramp(
                    kT_init=0.5, kT_final=1.5, n_steps=5e2, pressure=0.003
            )

    def test_temp_ramp_npt(self, pps_system):
        simulation = simulate.Simulation(
                pps_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.temp_ramp(
                kT_init=0.5, kT_final=1.5, n_steps=5e2, pressure=0.003
        )

    def test_temp_ramp_nvt(self, pps_system):
        simulation = simulate.Simulation(
                pps_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.temp_ramp(kT_init=0.5, kT_final=1.5, n_steps=5e2)

    def test_temp_ramp_nvt(self, pps_system):
        simulation = simulate.Simulation(
                pps_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.temp_ramp(kT_init=0.5, kT_final=1.5, n_steps=5e2)

    def test_bad_inputs(self, peek_system, cg_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                wall_axis=[1,0,0],
                mode="cpu"
        )

        # Pressure and walls quench
        with pytest.raises(ValueError):
            simulation.quench(
                kT=2,
                pressure=0.1,
                n_steps=5e2,
            )
        # Pressure and walls anneal
        with pytest.raises(ValueError):
            simulation.anneal(
                kT_init=2,
                kT_final=1,
                step_sequence = [5e2, 5e2],
                pressure=0.1,
            )
        # Coarse-grain and auto scale
        with pytest.raises(AssertionError):
            sim = simulate.Simulation(
                    system=cg_system,
                    auto_scale=True,
                    ref_values=None
            )

    def test_tree_nlist(self, peek_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu",
                nlist="Tree"
        )
        simulation.quench(
            kT=2, pressure=0.1, n_steps=5e2,
        )

    def test_quench_npt_no_shrink(self, peek_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.quench(
            kT=2, pressure=0.1, n_steps=5e2,
        )

    def test_quench_nvt_no_shrink(self, pps_system):
        simulation = simulate.Simulation(
                pps_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.quench(kT=2, n_steps=5e2)

    def test_anneal_npt_no_shrink(self, peek_system):
        simulation = simulate.Simulation(peek_system, tau_p=0.5, dt=0.0001, mode="cpu")
        simulation.anneal(
            kT_init=4, kT_final=2, pressure=0.0001, step_sequence=[5e2, 5e2]
        )

    def test_anneal_nvt_no_shrink(self, peek_system):
        simulation = simulate.Simulation(peek_system, dt=0.0001, mode="cpu")
        simulation.anneal(
            kT_init=4, kT_final=2, step_sequence=[5e2, 5e2]
        )

    def test_shrink_tree_nlist(self, peek_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        assert isinstance(
                simulation.sim.operations.integrator.forces[0].nlist,
                hoomd.md.nlist.Cell
        )
        simulation.shrink(
                n_steps=5e2,
                kT_init=2,
                kT_final=2,
                period=1,
                tree_nlist=True
        )
        assert isinstance(
                simulation.sim.operations.integrator.forces[0].nlist,
                hoomd.md.nlist.Cell
        )
        simulation.quench(kT=2, pressure=0.1, n_steps=5e2)

    def test_quench_npt_shrink(self, peek_system):
        simulation = simulate.Simulation(
                peek_system,
                tau_p=0.1,
                dt=0.0001,
                mode="cpu"
        )
        simulation.shrink(n_steps=5e2, kT_init=2, kT_final=2, period=1)
        simulation.quench(kT=2, pressure=0.1, n_steps=5e2)

    def test_quench_nvt_shrink(self, peek_system):
        simulation = simulate.Simulation(peek_system, dt=0.0001, mode="cpu")
        simulation.shrink(n_steps=5e2, kT_init=2, kT_final=2, period=1)
        simulation.quench(kT=2, n_steps=1e3)

    def test_anneal_npt_shrink(self, pekk_system):
        simulation = simulate.Simulation(
                pekk_system,
                dt=0.0001,
                tau_p=0.1,
                mode="cpu"
        )
        simulation.shrink(n_steps=5e2, kT_init=4, kT_final=4, period=1)
        simulation.anneal(
                kT_init=4,
                kT_final=2,
                pressure=0.1,
                step_sequence=[5e2, 5e2],
        )

    def test_shrink_ua(self, pps_system_noH):
        simulation = simulate.Simulation(pps_system_noH, mode="cpu")
        simulation.shrink(kT_init=2, kT_final=1, n_steps=5e2)

    def test_quench_ua(self, pekk_system_noH):
        simulation = simulate.Simulation(pekk_system_noH, mode="cpu")
        simulation.quench(kT=2, n_steps=5e2)

    def test_anneal_ua(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, mode="cpu")
        simulation.anneal(
            kT_init=4, kT_final=2, step_sequence=[5e2, 5e2],
        )

    def test_walls_x_quench(self, pekk_system_noH):
        simulation = simulate.Simulation(
                pekk_system_noH, dt=0.0003, wall_axis=[1,0,0], mode="cpu"
        )
        simulation.shrink(kT_init=4, kT_final=2, n_steps=1e3, period=5)
        simulation.quench(kT=2, n_steps=1e2)

    def test_walls_x_anneal(self, pps_system_noH):
        simulation = simulate.Simulation(
                pps_system_noH, dt=0.0003, wall_axis=[1,0,0], mode="cpu"
        )
        simulation.shrink(kT_init=4, kT_final=2, n_steps=1e3, period=5)
        simulation.anneal(kT_init=2, kT_final=4, step_sequence=[1e2, 1e2, 1e2])

    def test_walls_y_quench(self, peek_system_noH):
        simulation = simulate.Simulation(
                peek_system_noH, dt=0.0003, wall_axis=[0,1,0], mode="cpu"
        )
        simulation.shrink(kT_init=4, kT_final=2, n_steps=1e3, period=5)
        simulation.quench(kT=2, n_steps=1e2)

    def test_walls_y_anneal(self, peek_system_noH):
        simulation = simulate.Simulation(
                peek_system_noH, dt=0.0003, wall_axis=[0,1,0], mode="cpu"
        )
        simulation.shrink(kT_init=4, kT_final=2, n_steps=1e3, period=5)
        simulation.anneal(kT_init=2, kT_final=4, step_sequence=[1e2, 1e2, 1e2])

    def test_walls_z_quench(self, peek_system_noH):
        simulation = simulate.Simulation(
                peek_system_noH, dt=0.0003, wall_axis=[0,0,1], mode="cpu"
        )
        simulation.shrink(kT_init=4, kT_final=2, n_steps=1e3, period=5)
        simulation.quench(kT=2, n_steps=1e2)

    def test_walls_z_anneal(self, peek_system_noH):
        simulation = simulate.Simulation(
                peek_system_noH, dt=0.0003, wall_axis=[0,0,1], mode="cpu"
        )
        simulation.shrink(kT_init=4, kT_final=2, n_steps=1e3, period=5)
        simulation.anneal(kT_init=2, kT_final=4, step_sequence=[1e2, 1e2, 1e2])

    def test_weld_quench(self, test_interface_x):
        simulation = simulate.Simulation(
                test_interface_x, dt=0.0001, mode="cpu", wall_axis=[1,0,0]
        )
        simulation.quench(kT=2, n_steps=5e2)

    def test_weld_anneal(self, test_interface_y):
        simulation = simulate.Simulation(
                test_interface_y, dt=0.0001, mode="cpu", wall_axis=[0,1,0]
        )
        simulation.anneal(
            kT_init=4, kT_final=2, step_sequence=[5e2, 5e2, 5e2]
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

    def test_cg_sim_monomers(self, cg_system_monomers):
        simulation = simulate.Simulation(
                cg_system_monomers,
                r_cut=2.5,
                mode="cpu",
                ref_values = {"distance": 3.3997, "energy": 0.21, "mass": 15.99},
                cg_potentials_dir = "test_potentials"
        )
        simulation.quench(kT=3.5, n_steps=500)

    def test_cg_sim_components(self, cg_system_components):
        simulation = simulate.Simulation(
                cg_system_components,
                r_cut=2.5,
                mode="cpu",
                ref_values = {"distance": 3.3997, "energy": 0.21, "mass": 15.99},
                cg_potentials_dir = "test_potentials"
        )
        simulation.quench(kT=3.5, n_steps=500)

    def test_cg_sim_no_files(self, cg_system_monomers):
        with pytest.raises(RuntimeError):
            simulation = simulate.Simulation(
                    cg_system_monomers,
                    r_cut=2.5,
                    mode="cpu",
                    ref_values = {"distance": 3.3997, "energy": 0.21, "mass": 15.99},
                    cg_potentials_dir = "bad_dir"
            )
            simulation.quench(kT=3.5, n_steps=500)

    def test_quench_from_restart(self, pekk_system_noH, restart_gsd):
        simulation = simulate.Simulation(
                pekk_system_noH, dt=0.001, mode="cpu", restart=restart_gsd
        )
        simulation.quench(kT=2, n_steps=5e2)

    def test_anneal_from_restart(self, pekk_system_noH, restart_gsd):
        simulation = simulate.Simulation(
                pekk_system_noH, dt=0.001, mode="cpu", restart=restart_gsd
        )
        simulation.anneal(kT_init=2, kT_final=4, step_sequence = [5e2, 5e2])
