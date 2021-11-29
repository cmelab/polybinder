'''This script is uses uli_init to generate initial positions
   that are loaded in for the HTF_Online_Demo.py script.
   The output of this simulation must be renamed to correspond
   with the number of polymers and polymer sizes you wish to use.'''

import uli_init.simulate as simulate
import uli_init.system as system
import hoomd

system = system.System(molecule='PEEK', para_weight=1.0, system_type='pack',
                         density=0.8, n_compounds=[100],
                         polymer_lengths=[4], forcefield='gaff',
                         assert_dihedrals=True, remove_hydrogens=True)

sim = simulate.Simulation(system, gsd_write=1, mode='gpu', dt=0.0001)
sim.quench(kT=1., n_steps=1, shrink_steps=1, shrink_kT=1.0, shrink_period=1)
