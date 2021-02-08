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

# building a HTF model for coarse graining
# here use the single-molecule file for simplicity
# TODO: update to get one molecule from a .gsd with multiple
#       e.g. grab first entry in htf.find_molecules(system) -> do the rest
fname = '1-length-4-peek-para-only.gsd'
gsdfile = gsd.hoomd.open(fname)
context = hoomd.context.initialize('--mode=cpu')
system = hoomd.init.read_gsd(filename=fname)
context.sorter.disable()

molecule_mapping_index = htf.find_molecules(system)

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

fname = '100-length-4-peek-para-only-production.gsd'
gsdfile = gsd.hoomd.open(fname)
context = hoomd.context.initialize('--mode=cpu')
system = hoomd.init.read_gsd(filename=fname)
context.sorter.disable()
set_rcut = 11.0
molecule_mapping_index = htf.find_molecules(system)

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
univ = mda.Universe(fname)

# create an edge list
beads_per_molecule = 12
bonds_per_molecule = beads_per_molecule - 1 # linear polymer
bonds_matrix = np.zeros([bonds_per_molecule * M, 2])
#print(M)
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

a = nx.Graph(adjacency_matrix)
#print('cg_mapping IS THIS --> ', cg_mapping)
#print('adjmat IS THIS --> ', adjacency_matrix)
b = dict(nx.all_pairs_shortest_path_length(a))
# print(b[12]) # this DOES have the key 12
class TrajModel(htf.SimModel):
    def setup(self, cg_num, adjacency_matrix, CG_NN, cg_mapping, rcut):
        self.cg_num = cg_num
        self.adjacency_matrix = adjacency_matrix
        self.CG_NN = CG_NN
        self.cg_mapping = cg_mapping
        self.rcut = rcut
        self.avg_cg_rdf = tf.keras.metrics.MeanTensor() # set up CG RDF tracking
    def compute(self, nlist, positions, box):
        # calculate the center of mass of a CG bead
        box_size = htf.box_size(box) # [16., 16., 16.]
        mapped_pos = htf.center_of_mass(positions=positions[:,:3],
                                        mapping=self.cg_mapping, 
                                        box_size= box_size)
        print('made it past to mapped_pos')
        #print(self.adjacency_matrix)
        # print(
        #     dict(
        #         nx.all_pairs_shortest_path_length(
        #             nx.Graph(
        #                 self.adjacency_matrix
        #             )
        #         )
        #     ).keys()
        # )
        cg_graph = htf.compute_cg_graph(DSGPM=False,
                                       infile=None,
                                       adj_mat=self.adjacency_matrix,
                                       cg_beads=self.cg_num)
        # create mapped neighbor list
        mapped_nlist = htf.compute_nlist(mapped_pos, self.rcut, self.CG_NN, box_size, True)
        # compute RDF for mapped particles
        cg_rdf = htf.compute_rdf(mapped_nlist, [0.1, self.rcut])
        self.avg_cg_rdf.update_state(cg_rdf)
        return mapped_pos, cg_graph, box, box_size
nneighbor_cutoff = 32
model = TrajModel(nneighbor_cutoff,
                 cg_num=12,
                 adjacency_matrix=adjacency_matrix,
                 CG_NN=nneighbor_cutoff,
                 cg_mapping=cg_mapping,
                 output_forces=False,
                 rcut=set_rcut,
                 check_nlist=False)
#print(adjacency_matrix)


    
def make_frame(i, positions):
    s = gsd.hoomd.Snapshot()
    s.configuration.box = [16., 16., 16., 0., 0., 0.]
    s.configuration.step = i
    s.particles.N = beads_per_molecule * M
    s.particles.position = positions
    s.bonds.N = bonds_per_molecule * M
    s.bonds.group = bonds_matrix
    #print(s)
    return s

write_CG_traj = False

if write_CG_traj:
    print(f'Applying CG mapping to {fname}')
    f = gsd.hoomd.open(name=f'CG_traj-{fname}', mode='wb+')
    i = 0
for inputs, ts in htf.iter_from_trajectory(nneighbor_cutoff, univ, r_cut=set_rcut):
    if i % 100 == 0:
        print(f'made it to step {i:04d}', end='\r')
    result = model(inputs)
    #print('    cg_model is: ', result[1], end='\r')
    particle_positions = np.array(result[0])
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