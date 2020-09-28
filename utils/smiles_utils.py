import json
import os
import deepsmiles
import mbuild as mb



def convert_smiles(smiles=False, deep=False):
    '''
    smiles and deep must be str format
    Converts from SMILES to DeepSMILES and vice versa.
    Whichever has a string provided, will convert to the other.
    If strings are proivded for both, then nothing happens
    '''
    converter = deepsmiles.Converter(rings=True, branches=True)
    if smiles and deep:
        print('Only provide a string for one of smiles or deep')
        return()
    if smiles: # Convert from SMILES to DeepSMILES
        deep_string = converter.encode(smiles)
        return deep_string
    if deep: # Convert from DeepSMILES to SMILES
        smiles_string = converter.decode(deep)
        return smiles_string

def viz(smiles_string, deep=True, energy_min=False):
    '''
    Uses mbuild to visualize a given smiles string.
    Accepts either smiles string or Deepsmiles string.
    If smiles_string is in Deepsmiles format, then deep = True
    energy_min can perform mbuild's energy minimization if True
    '''
    if deep:
        smiles_string = convert_smiles(deep = smiles_string)
    comp = mb.load(smiles_string, smiles = True)
    if energy_min:
        comp.energy_minimize(algorithm='md', steps=100)
    comp.visualize().show()


def read_comp(fpath):
    f = open(os.path.join('typed-components', fpath))
    data = json.load(f)
    f.close()
    return data