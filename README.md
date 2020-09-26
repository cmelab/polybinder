# uli-init
Initialization of thermoplastic polymer systems to simulate thermal welding

## Installation

Installation of uli-init requires the conda package manager

1. Clone this repository:  

```
git clone git@github.com:chrisjonesBSU/uli-init.git  
cd uli-init  
```

2. Set up and activate environment:  

```
conda env create -f environment.yml  
conda activate uli
```  

3. Install a fork of mBuild:  

```
git clone git@github.com:chrisjonesBSU/mbuild.git  
cd mbuild  
git checkout uli  
pip install -e .  
```

