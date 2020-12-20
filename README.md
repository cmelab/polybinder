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

```
conda env create -f environment.yml  
conda activate uli
```  
### 3. Install the HOOMD-blue molecular dynamics package: ###  

```
conda install -c conda-forge hoomd
```  

***OR*** 

To configure hoomd to run on GPUs, following the installation instructions found in the [hoomd docs](https://hoomd-blue.readthedocs.io/en/stable/installation.html)

### 4. Install this repository ###

```
pip install -e .
```

### 5. Install the GAFF-Foyer repository ###
```
cd path-to-your-repos
git clone git@github.com:rsdefever/GAFF-foyer.git
cd  GAFF-foyer
pip install -e .
```

## Basic Usage

So far, essentially all of the functionality lives in the `simulate.py` file

`from uli_init import simulate`

There are two primary classes used to initialize a system and simulation:  

`simulate.System()`
Used to generate the entire system, and ultimately a paramaterized parmed struture object.

`simulate.Simulation()`
This class takes in the parmed structure from `System` and initializes one of two possible simulation methods using the Hoomd package

1. quench `Simulation.quench()`
2. anneal `Simulation.anneal()`
