import numpy as np
import matplotlib.pyplot as plt
import hoomd
import gsd
import glob

param_fnames = sorted(glob.glob("*params.txt"))
print(param_fnames)

# plot our parameters over time
for fname in param_fnames:
    params = np.genfromtxt(fname)
    short_name = fname.split('.')[0] # discard file extension
    # first dimension is timesteps
    plt.figure()
    for i, param in enumerate(params[0]):
        plt.plot(range(params.shape[0]), params[:,i], label=f'{short_name}[{i}]')
        plt.legend()
    plt.xlabel('Steps')
    plt.ylabel(short_name)
    plt.savefig(f'{short_name}.png')
    
M = 100
# create an edge list
beads_per_molecule = 12
bonds_per_molecule = beads_per_molecule - 1 # linear polymer
bonds_matrix = np.zeros([bonds_per_molecule * M, 2]) # TODO: read the number, not hard-code 100
bonds_matrix[0][1] = 1
offset = 0

# this puts the indices of bonded beads as pairs 
# i.e. the edge list of a graph
for i in range(1, bonds_matrix.shape[0]):
    bonds_matrix[i][0] = i + i//11
    bonds_matrix[i][1] = i+1 + i//11


# for writing out the CG trajectory
def make_frame(i, positions, bonds_matrix):
    s = gsd.hoomd.Snapshot()
    s.configuration.box = [16., 16., 16., 0., 0., 0.]
    s.configuration.step = i
    s.particles.N = beads_per_molecule * M
    s.particles.position = positions
    s.bonds.N = bonds_per_molecule * M
    s.bonds.group = bonds_matrix
    return s

angle_params = np.genfromtxt(param_fnames[0])
bond_params = np.genfromtxt(param_fnames[1])
dihedral_params = np.genfromtxt(param_fnames[2])
lj_params = np.genfromtxt(param_fnames[3])

positions = np.load('cg_positions.npy')
positions = positions[:,:,:3] # remove types

gsdfile = gsd.hoomd.open(name=f'cg_trajectory.gsd', mode='wb+')

for i in range(positions.shape[0]):
    this_pos = positions[i]
    frame = make_frame(i, this_pos, bonds_matrix)
    gsdfile.append(frame)