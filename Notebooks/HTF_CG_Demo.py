import os
#os.environ['CUDA_VISIBLE_DEVICES'] =  '-1'
import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers
import hoomd
import hoomd.md
import hoomd.htf as htf
import numpy as np
import gsd
import gsd.hoomd
import pickle
import matplotlib.pyplot as plt
import os

# building a HTF model for coarse graining
# here use the single-molecule file for simplicity
# TODO: save out the CG Mapping to file, load it in
# TODO: break this file up into a few scripts
# e.g. one for making mapping, one for running from traj...
# TODO: once saving is set, get pipeline for starting a sim fresh with uli-init
# TODO: set up CG FF NN with all the molecule bond/angle/dihedral/LJ parameters

def get_mol_mapping_idx(filename):
    '''Takes a filename of a .gsd file WITHOUT the '.gsd', loads that gsd,
    then loads or creates a molecule mapping index from it.'''
    # if the mapping doesn't exist, make it
    gsdfile = gsd.hoomd.open(f'{filename}.gsd')
    context = hoomd.context.initialize('--mode=cpu')
    system = hoomd.init.read_gsd(filename=f'{filename}.gsd')
    context.sorter.disable()
    if not os.path.exists(f'{filename}-mapping.npy'):
        molecule_mapping_index = htf.find_molecules(system)
        np.save(f'{filename}-mapping.npy', np.array(molecule_mapping_index))
    # if it does, load from it instead    
    else:
        molecule_mapping_index = np.load(f'{filename}-mapping.npy')
    return system, molecule_mapping_index

system, molecule_mapping_index = get_mol_mapping_idx('1-length-4-peek-para-only')

graph = nx.Graph()
# add all our particles and bonds
for particle in system.particles:
    graph.add_node(particle.tag, name=particle.type)
for bond in system.bonds:
    graph.add_edge(bond.a, bond.b)
plt.figure()
plt.title('BEFORE')
nx.draw(graph, with_labels=True)
plt.savefig('before.png')
# judiciously snip bonds
degrees_dict = dict(graph.degree())

for i, bond in enumerate(system.bonds):
    if bond.type == 'c-ca' or bond.type == 'ca-c':
        if degrees_dict[bond.a] == 3 and degrees_dict[bond.b] == 3:
            graph.remove_edge(bond.a, bond.b)
    elif bond.type == 'ca-os' or bond.type == 'os-ca':
        if degrees_dict[bond.a] == 2 and degrees_dict[bond.b] == 3 or\
        degrees_dict[bond.a] == 3 and degrees_dict[bond.b] == 2:
            graph.remove_edge(bond.a, bond.b)
    degrees_dict = dict(graph.degree())

subgraph_list = list(nx.connected_components(graph))
plt.figure()
plt.title('AFTER')
nx.draw(graph, with_labels=True)
plt.savefig('after.png')

# now we have our beads grouped up, we need to get their mapping
# get total N atoms
N = sum([len(m) for m in molecule_mapping_index])
# get molecule count
M = len(molecule_mapping_index)
# atoms per molecule
MN = len(molecule_mapping_index[0])
print('N_atoms:', N,'\nN_molecules:', M,'\nN_atoms_per_molecule:', MN)
# make sure we didn't miss any particles
assert(sum([len(item) for item in subgraph_list]) == MN)

# create a mapping for our molecules
# these are 4-monomer polymers, and we're doing 3 beads per monomer
# therefore, we need a 12 x 88 matrix


mapping_arr = np.zeros((12,MN))

for i, subgraph in enumerate(subgraph_list):
    for atom_idx in subgraph:
        mapping_arr[i][atom_idx] = 1
        
N = sum([len(m) for m in molecule_mapping_index])
# get molecule count
M = len(molecule_mapping_index)
# atoms per molecule
MN = len(molecule_mapping_index[0])
# again make sure we didn't miss any atoms
assert(np.sum(mapping_arr) == MN)

bead_number = mapping_arr.shape[0]

set_rcut = 11.0
system, molecule_mapping_index = get_mol_mapping_idx('100-length-4-peek-para-only-production')

cg_mapping = htf.sparse_mapping([mapping_arr for _ in molecule_mapping_index],
                               molecule_mapping_index, system=system)
N = sum([len(m) for m in molecule_mapping_index])
# get molecule count
M = len(molecule_mapping_index)
# atoms per molecule
MN = len(molecule_mapping_index[0])
print('N_atoms:', N,'\nN_molecules:', M,'\nN_atoms_per_molecule:', MN)
assert cg_mapping.shape == (M * bead_number, N)

import MDAnalysis as mda
univ = mda.Universe('100-length-4-peek-para-only-production.gsd')

# create an edge list
beads_per_molecule = 12
bonds_per_molecule = beads_per_molecule - 1 # linear polymer
bonds_matrix = np.zeros([bonds_per_molecule * M, 2])
bonds_matrix[0][1] = 1
offset = 0

# this puts the indices of bonded beads as pairs 
# i.e. the edge list of a graph
for i in range(1, bonds_matrix.shape[0]):
    bonds_matrix[i][0] = i + i//11
    bonds_matrix[i][1] = i+1 + i//11

# make adjacency matrix (N_beads x N_beads)
# adj_mat[i][j] = 0 if beads (i, j) not bonded, 1 if they are
adjacency_matrix = np.zeros([M * 12, M * 12])
for pair in bonds_matrix:
    i, j = int(pair[0]), int(pair[1])
    adjacency_matrix[i][j] = adjacency_matrix[j][i] = 1

class TrajModel(htf.SimModel):
    def setup(self, cg_num, adjacency_matrix, CG_NN, cg_mapping, rcut):
        self.cg_num = cg_num
        self.adjacency_matrix = adjacency_matrix
        self.CG_NN = CG_NN
        self.cg_mapping = cg_mapping
        self.rcut = rcut
        self.avg_cg_rdf = tf.keras.metrics.MeanTensor() # set up CG RDF tracking

        self.avg_cg_radii = tf.keras.metrics.MeanTensor()
        self.avg_cg_angles = tf.keras.metrics.MeanTensor()
        self.avg_cg_dihedrals = tf.keras.metrics.MeanTensor()

    def compute(self, nlist, positions, box):
        # calculate the center of mass of a CG bead
        box_size = htf.box_size(box) # [16., 16., 16.]
        mapped_pos = htf.center_of_mass(positions=positions[:,:3],
                                        mapping=self.cg_mapping, 
                                        box_size= box_size)
        cg_features = htf.compute_cg_graph(DSGPM=False,
                                       infile=None,
                                       adj_mat=self.adjacency_matrix,
                                       cg_beads=self.cg_num)

        radii_tensor = []
        angles_tensor = []
        dihedrals_tensor = []

        # because these are tensors, can't use list comprehension
        for i in range(len(cg_features[0])):
            cg_radius = htf.mol_bond_distance(CG=True,
                                              cg_positions=mapped_pos,
                                              b1=cg_features[0][i][0],
                                              b2=cg_features[0][i][1]
                                              )
            radii_tensor.append(cg_radius)
        self.avg_cg_radii.update_state(radii_tensor)

        for j in range(len(cg_features[1])):
            cg_angle = htf.mol_angle(CG=True,
                                     cg_positions=mapped_pos,
                                     b1=cg_features[1][j][0],
                                     b2=cg_features[1][j][1],
                                     b3=cg_features[1][j][2]
                                     )
            angles_tensor.append(cg_angle)
        self.avg_cg_angles.update_state(angles_tensor)

        for k in range(len(cg_features[2])):
            cg_dihedral = htf.mol_dihedral(CG=True,
                                           cg_positions=mapped_pos,
                                           b1=cg_features[2][k][0],
                                           b2=cg_features[2][k][1],
                                           b3=cg_features[2][k][2],
                                           b4=cg_features[2][k][3],
                                           )
            dihedrals_tensor.append(cg_dihedral)
        self.avg_cg_dihedrals.update_state(dihedrals_tensor)

        # create mapped neighbor list
        mapped_nlist = htf.compute_nlist(mapped_pos, self.rcut, self.CG_NN, box_size, True)
        # compute RDF for mapped particles
        cg_rdf = htf.compute_rdf(mapped_nlist, [0.1, self.rcut])
        self.avg_cg_rdf.update_state(cg_rdf)
        return mapped_pos, cg_features, radii_tensor, angles_tensor, dihedrals_tensor, box, box_size

nneighbor_cutoff = 32
model = TrajModel(nneighbor_cutoff,
                 cg_num=12, # beads per molecule, not total
                 adjacency_matrix=adjacency_matrix,
                 CG_NN=nneighbor_cutoff,
                 cg_mapping=cg_mapping,
                 output_forces=False,
                 rcut=set_rcut,
                 check_nlist=False)
    
# for writing out the CG trajectory
def make_frame(i, positions):
    s = gsd.hoomd.Snapshot()
    s.configuration.box = [16., 16., 16., 0., 0., 0.]
    s.configuration.step = i
    s.particles.N = beads_per_molecule * M
    s.particles.position = positions
    s.bonds.N = bonds_per_molecule * M
    s.bonds.group = bonds_matrix
    return s

write_CG_traj = False

avg_bond_lengths = []
avg_bond_angles = []
avg_dihedral_angles = []

if write_CG_traj:
    print(f'Applying CG mapping to {fname}')
    f = gsd.hoomd.open(name=f'CG_traj-{fname}', mode='wb+')
i = 0
for inputs, ts in htf.iter_from_trajectory(nneighbor_cutoff, univ, r_cut=set_rcut):
    if i == 100:
        break
    result = model(inputs)
    particle_positions = np.array(result[0])
    avg_bond_lengths.append(result[2])
    avg_bond_angles.append(result[3])
    avg_dihedral_angles.append(result[4])
    if write_CG_traj:
        f.append(make_frame(i, particle_positions))
    i+=1

cg_rdf = model.avg_cg_rdf.result().numpy()

plt.figure()
plt.plot(cg_rdf[1,:], cg_rdf[0,:], label='Mapped (CG)')
plt.xlabel('r [$\AA$]')
plt.ylabel('$g(r)$')
plt.legend()
plt.savefig('CG_RDF.svg')

# plot average CG bond radii
cg_radii = model.avg_cg_radii.result().numpy()
print(cg_radii.shape)

np.save('cg_radii.npy', np.array(avg_bond_lengths))

# plt.figure()
# plt.hist(np.array(avg_bond_lengths))
# plt.xlabel('r [$\AA$]')
# plt.ylabel('Count')
# plt.title('CG Bond Length Histogram')
# plt.legend()
# plt.savefig('CG_Radii.svg')

# plot average CG bond angles
# cg_radii = model.avg_cg_angles.result().numpy()

np.save('cg_angles.npy', np.array(avg_bond_angles))

# plt.figure()
# plt.plot(cg_radii[1,:], cg_radii[0,:], label='Mapped (CG)')
# plt.xlabel('$\theta$ [Degrees(?)]')
# plt.ylabel('$g(r)$?')
# plt.legend()
# plt.savefig('CG_Angles.svg')

# plot average CG dihedral angles
# cg_radii = model.avg_cg_dihedrals.result().numpy()

np.save('cg_dihedrals.npy', np.array(avg_dihedral_angles))

# plt.figure()
# plt.plot(cg_radii[1,:], cg_radii[0,:], label='Mapped (CG)')
# plt.xlabel('$\phi$ [Degrees?]')
# plt.ylabel('$g(r)$?')
# plt.legend()
# plt.savefig('CG_Radii.svg')