from uli_init import simulate
from uli_init.tests.base_test import BaseTest

class TestSimulate(BaseTest):

    def test_quench(self, peek_system):
        simulation = simulate.Simulation(peek_system, dt=0.0001, mode='cpu')
        simulation.quench(kT=2, n_steps=1e3, shrink_steps=1e3, walls=False)

    def test_anneal(self, pekk_system):
        simulation = simulate.Simulation(pekk_system, dt = 0.0001, mode='cpu')
        simulation.anneal(kT_init=4, kT_final=2,
                            step_sequence=[1e3, 1e3],
                            shrink_steps=1e3,
                            walls=False)

    def test_quench_noH(self, pekk_system_noH):
        simulation = simulate.Simulation(pekk_system_noH, dt=0.001, mode='cpu')
        simulation.quench(kT=2, n_steps=1e3, shrink_steps=1e3, walls=False)

    def test_anneal_noH(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt = 0.001, mode='cpu')
        simulation.anneal(kT_init=4, kT_final=2,
                            step_sequence=[1e3, 1e3, 1e3],
                            shrink_steps=1e3,
                            walls=False)
   
    def test_walls_quench(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt = 0.001, mode='cpu')
        simulation.quench(kT=2, n_steps=1e3, shrink_steps=1e3, walls=True)

    def test_walls_anneal(self, peek_system_noH):
        simulation = simulate.Simulation(peek_system_noH, dt = 0.001, mode='cpu')
        simulation.anneal(kT_init=4, kT_final=2,
                        step_sequence=[1e3, 1e3],
                        shrink_steps=1e3,
                        walls=True)
