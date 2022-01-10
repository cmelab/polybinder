[![pytest](https://github.com/cmelab/polybinder/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/polybinder/actions/workflows/pytest.yml)
[![build_cpu](https://github.com/cmelab/polybinder/actions/workflows/build_cpu.yml/badge.svg)](https://github.com/cmelab/polybinder/actions/workflows/build_cpu.yml)
[![build_gpu](https://github.com/cmelab/polybinder/actions/workflows/build_gpu.yml/badge.svg)](https://github.com/cmelab/polybinder/actions/workflows/build_gpu.yml)
[![codecov](https://codecov.io/gh/cmelab/polybinder/branch/master/graph/badge.svg?token=8Z9MBA7M16)](https://codecov.io/gh/cmelab/polybinder)
# PolyBinder 
Initialization of thermoplastic polymer systems to simulate thermal welding

## Installation

Installation of PolyBinder requires the conda package manager

### 1. Clone this repository: ###  

```
git clone git@github.com:cmelab/polybinder.git  
cd polybinder  
```

### 2. Set up and activate environment: ###  
#### a. Using HOOMD-blue from conda:
```
conda env create -f environment-cpu.yml  
conda activate polybinder 
python -m pip install .
```  

## Containers
PolyBinder containers with HOOMD compiled to work on cpu and gpu are availiable: `polybinder_cpu` and `polybinder_gpu`. 

To use PolyBinder in a prebuilt container (using Singularity), run:
```
singularity pull docker://cmelab/polybinder_cpu:latest
singularity exec polybinder_cpu_latest.sif bash
```
Or using Docker, run:
```
docker pull cmelab/polybinder_cpu:latest
docker run -it cmelab/polybinder_cpu:latest
```

## Basic Usage

Essentially all of the functionality lives in the `simulate.py` and `system.py` files

`from polybinder import simulate, system`

There are two primary classes used to initialize a system and simulation:  

`system.System()`
Used to generate the entire system, and ultimately a paramaterized parmed struture object.

`simulate.Simulation()`
This class takes in the parmed structure from `System` and initializes one of two possible simulation methods using the Hoomd package

1. quench `Simulation.quench()`
2. anneal `Simulation.anneal()`
