[![pytest](https://github.com/cmelab/uli-init/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/uli-init/actions/workflows/pytest.yml)
[![build_cpu](https://github.com/cmelab/uli-init/actions/workflows/build_cpu.yml/badge.svg)](https://github.com/cmelab/uli-init/actions/workflows/build_cpu.yml)
[![build_gpu](https://github.com/cmelab/uli-init/actions/workflows/build_gpu.yml/badge.svg)](https://github.com/cmelab/uli-init/actions/workflows/build_gpu.yml)
[![codecov](https://codecov.io/gh/cmelab/uli-init/branch/master/graph/badge.svg?token=8Z9MBA7M16)](https://codecov.io/gh/cmelab/uli-init)
# uli-init
Initialization of thermoplastic polymer systems to simulate thermal welding

## Installation

Installation of uli-init requires the conda package manager

### 1. Clone this repository: ###  

```
git clone git@github.com:cmelab/uli-init.git  
cd uli-init  
```

### 2. Set up and activate environment: ###  
#### a. Using HOOMD-blue from conda:
```
conda env create -f environment.yml  
conda activate uli-init
```  

## Containers
Uli-Init containers with HOOMD compiled to work on cpu and gpu are availiable: `uliinit_cpu` and `uliinit_gpu`. 

To use uli-init in a prebuilt container (using Singularity), run:
```
singularity pull docker://cmelab/uliinit_cpu:latest
singularity exec ulitinit_cpu_latest.sif bash
```
Or using Docker, run:
```
docker pull cmelab/uliinit_cpu:latest
docker run -it cmelab/uniinit_cpu:latest
```

## Basic Usage

Essentially all of the functionality lives in the `simulate.py` and `system.py` files

`from uli_init import simulate, system`

There are two primary classes used to initialize a system and simulation:  

`system.System()`
Used to generate the entire system, and ultimately a paramaterized parmed struture object.

`simulate.Simulation()`
This class takes in the parmed structure from `System` and initializes one of two possible simulation methods using the Hoomd package

1. quench `Simulation.quench()`
2. anneal `Simulation.anneal()`
