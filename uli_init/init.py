#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""

import signac
import logging
from collections import OrderedDict
from itertools import product

def get_parameters():
    '''
    DOCS ROUGH DRAFT:
    
    All temperature parameters are entered as reduced temperature units

    If you want to use pdi and M_n:
        Comment out n_compounds and polymer_lengths lines

    If you only want to run a quench simulation
        Comment out kT_anneal, anneal_sequence lines

    If you only want to run an anneal simulation
        Comment out kT_quench and n_steps lines

    Don't forget to change the name of the project
    '''
    parameters = OrderedDict()
    # System generation parameters:
    parameters["molecule"] = ['PEEK', 'PEKK']
    parameters["para_weight"] = [0.60, 0.70, 0.80]
    parameters["density"] = [0.9, 1.0, 1.2, 1.3]
    parameters["n_compounds"] = [[50, 75, 40], # List of lists 
                                 [100, 150, 80], 
                                 [200, 300, 160]
                                ]
    parameters["polymer_lengths"] = [[5, 10, 15] # List of lists
                                    ]            # Must match length of n_compound lists
    parameters["pdi"] = [None]
    parameters["M_n"] = [None]
    parameters["forcefield"] = ['gaff']
    parameters["remove_hydrogens"] = [False]
    
    # Simulation parameters
    parameters["tau"] = [0.1]
    parameters["dt"] = [0.0001]
    parameters["e_factor"] = [0.5]
    parameters["procedure"] = ["quench", "anneal"]
        # Quench related params:
    parameters["kT_quench"] = [1.0] # Reduced Temp
    parameters["n_steps"] = [1e7]
        # Anneal related params
    parameters["kT_anneal"] = [[2.0, 1.0]
                              ] # List of [initial kT, final kT] Reduced Temps
    parameters["anneal_sequence"] = [[1e6, 3e5, 3e5, 5e5, 5e5, 1e6] # List of lists (n_steps)
                                    ]
    parameters["schedule"] = [None]


def main():
    project = signac.init_project("project")
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("steps", parent_statepoint["n_steps"])
    project.write_statepoints()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
