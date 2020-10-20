# uli-init
Initialization of thermoplastic polymer systems to simulate thermal welding

## Installation

Installation of uli-init requires the conda package manager

### 1. Clone this repository: ###  

```
git clone git@github.com:chrisjonesBSU/uli-init.git  
cd uli-init  
```

### 2. Set up and activate environment: ###  

```
conda env create -f environment.yml  
conda activate uli
```  

***OR*** 

If you want to install the MoSDeF packages (foyer, mbuild, etc..) from source then use the `environment-dev.yml` file
```
conda env create -f environment-dev.yml
conda activate uli-dev
```

Then `pip install -e .` from within each MoSDeF repository. Right now foyer, mBuild, and GAFF_Foyer are required.  

	- https://github.com/mosdef-hub/foyer  

	- https://github.com/mosdef-hub/mbuild  

	- https://github.com/rsdefever/GAFF-foyer.git


### 3. Install the HOOMD-blue molecular dynamics package: ###  

`conda install -c conda-forge hoomd`  

***OR*** 

To configure hoomd to run on GPUs, following the installation instructions found in the [hoomd docs](https://hoomd-blue.readthedocs.io/en/stable/installation.html)

### 4. Install this repository ###

```
pip install -e .
```
