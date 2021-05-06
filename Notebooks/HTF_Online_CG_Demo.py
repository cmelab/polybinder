import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '-1'
import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers
import hoomd
import hoomd.md
import hoomd.htf as htf
import numpy as np
import gsd
import os
from uli_init.utils.smiles_utils import viz
import uli_init.simulate as simulate

# building an HTF model for coarse graining

def get_mol_mapping_idx(filename):
    '''Takes a filename of a .gsd file WITHOUT the '.gsd', loads that gsd,
    then loads or creates a molecule mapping index from it.'''
    # if the mapping doesn't exist, make it
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

# Steps to do HOOMD-TF:
# 1) build HOOMD-TF keras model
# 2) compile keras model
# 3) create tfcompute object from model
# 4) set up HOOMD simulation (uli-init code)
# 5) pass in model to uli-init Simulation object
# 7) run the hoomd simulation (call the quench method)

one_molecule_fname = '1-length-4-peek-para-only'
system, molecule_mapping_index = get_mol_mapping_idx(one_molecule_fname)

graph = nx.Graph()
# add all our particles and bonds
for particle in system.particles:
    graph.add_node(particle.tag, name=particle.type)
for bond in system.bonds:
    graph.add_edge(bond.a, bond.b)
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

set_rcut = 7.0
fname = '100-length-4-peek-para-only-production'
system, molecule_mapping_index = get_mol_mapping_idx(fname)

cg_mapping = htf.sparse_mapping([mapping_arr for _ in molecule_mapping_index],
                               molecule_mapping_index, system=system)
N = sum([len(m) for m in molecule_mapping_index])
# get molecule count
M = len(molecule_mapping_index)
# atoms per molecule
MN = len(molecule_mapping_index[0])
print('N_atoms:', N,'\nN_molecules:', M,'\nN_atoms_per_molecule:', MN)
assert cg_mapping.shape == (M * bead_number, N)

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

# create Lennard-Jones energy-calculating layer
class LJLayer(tf.keras.layers.Layer):
    def __init__(self, sigma, epsilon):
        super().__init__(self, name='lj')
        self.start_vals = [sigma, epsilon]
        self.w = self.add_weight(
            shape=[2],
            initializer=tf.constant_initializer([sigma, epsilon]),
            constraint=tf.keras.constraints.NonNeg()
        )
    # call takes only particle radii (pass in from neighbor list)
    # returns energy contribution from LJ interactions
    def call(self, r):
        r6 = tf.math.divide_no_nan(self.w[0]**6, r**6)
        energy = self.w[1] * 4.0 * (r6**2 - r6)
        # divide by 2 to avoid double-counting
        return energy / 2.

class BondLayer(tf.keras.layers.Layer):
    # harmonic bond potential
    def __init__(self, k_b, r0):
        # we only have one bond type, so we only need one k & r0
        super().__init__(self, name='bonds')
        # set initial values for bond spring constant (k_b) and equilibrium length (r0)
        self.start = [k_b, r0]
        self.w = self.add_weight(
            shape=[2],
            initializer=tf.constant_initializer([k_b, r0]),
            constraint=tf.keras.constraints.NonNeg()
        )
        
    def call(self, r):
        energy = self.w[0] * (r - self.w[1])**2
        # don't divide by 2 here because we are doing per-bond (not neighbor-list-based)
        return energy

class AngleLayer(tf.keras.layers.Layer):
    # harmonic angle potential
    def __init__(self, k_a, theta0):
        # only one angle type, so we only need one k & theta0
        super().__init__(self, name='angles')
        # set initial values for angle spring constant (k) and equilibrium theta
        self.start = [k_a, theta0]
        self.w = self.add_weight(
            shape=[2],
            initializer=tf.constant_initializer([k_a, theta0]),
            constraint=tf.keras.constraints.NonNeg()
        )
    def call(self, theta):
        energy = self.w[0] * (theta - self.w[1])**2
        return energy

class DihedralLayer(tf.keras.layers.Layer):
    # harmonic cosine potential
    def __init__(self, k_d, phi0):
        # only one type of dihedral, so we only need one k & phi0
        super().__init__(self, name='dihedrals')
        # set initial values for dihedral spring constant (k) and equilibrium phi
        self.start = [k_d, phi0]
        self.w = self.add_weight(
            shape=[2],
            initializer=tf.constant_initializer([k_d, phi0]),
            constraint=tf.keras.constraints.NonNeg()
        )
    def call(self, phi):
        energy = self.w[0] * (tf.math.cos(phi) - tf.math.cos(self.w[1]))**2
        return energy

class TrajModel(htf.SimModel):
    def setup(self, cg_num, adjacency_matrix, CG_NN, rcut):
        self.cg_num = cg_num
        self.adjacency_matrix = adjacency_matrix
        self.CG_NN = CG_NN
        #self.cg_mapping = cg_mapping
        self.rcut = rcut
        self.avg_cg_rdf = tf.keras.metrics.MeanTensor() # set up CG RDF tracking

        self.avg_cg_radii = tf.keras.metrics.MeanTensor()
        self.avg_cg_angles = tf.keras.metrics.MeanTensor()
        self.avg_cg_dihedrals = tf.keras.metrics.MeanTensor()

        # energy layers
        self.lj_energy = LJLayer(1., 1.)
        # just a guess at bond length
        self.bond_energy = BondLayer(1., 2.)
        self.angle_energy = AngleLayer(1., 3.14/2.)
        self.dihedral_energy = DihedralLayer(1., 3.14/2.)

    def compute(self, nlist, positions, box):
        # calculate the center of mass of a CG bead
        box_size = htf.box_size(box) # [16., 16., 16.]
        cg_features = htf.compute_cg_graph(DSGPM=False,
                                       infile=None,
                                       adj_mat=self.adjacency_matrix,
                                       cg_beads=self.cg_num)

        radii_list = []
        angles_list = []
        dihedrals_list = []

        # because these are tensors, can't use list comprehension
        for i in range(len(cg_features[0])):
            cg_radius = htf.mol_bond_distance(CG=True,
                                              cg_positions=positions[:,:3],
                                              b1=cg_features[0][i][0],
                                              b2=cg_features[0][i][1],
                                              box=box
                                              )
            radii_list.append(cg_radius)
        self.avg_cg_radii.update_state(radii_list)

        for j in range(len(cg_features[1])):
            cg_angle = htf.mol_angle(CG=True,
                                     cg_positions=positions[:,:3],
                                     b1=cg_features[1][j][0],
                                     b2=cg_features[1][j][1],
                                     b3=cg_features[1][j][2],
                                     box=box
                                     )
            angles_list.append(cg_angle)
        self.avg_cg_angles.update_state(angles_list)

        for k in range(len(cg_features[2])):
            cg_dihedral = htf.mol_dihedral(CG=True,
                                           cg_positions=positions[:,:3],
                                           b1=cg_features[2][k][0],
                                           b2=cg_features[2][k][1],
                                           b3=cg_features[2][k][2],
                                           b4=cg_features[2][k][3],
                                           box=box
                                           )
            dihedrals_list.append(cg_dihedral)
        self.avg_cg_dihedrals.update_state(dihedrals_list)

        # create mapped neighbor list
        mapped_nlist = nlist# htf.compute_nlist(mapped_pos, self.rcut, self.CG_NN, box_size, True)
        # compute RDF for mapped particles
        cg_rdf = htf.compute_rdf(mapped_nlist, [0.1, self.rcut])
        self.avg_cg_rdf.update_state(cg_rdf)

        # now calculate our total energy and train
        nlist_r = htf.safe_norm(tensor=mapped_nlist[:, :, :3], axis=2)
        lj_energy = self.lj_energy(nlist_r) # TODO: something is going on with these  indices.
        lj_energy_total = tf.reduce_sum(input_tensor=lj_energy, axis=1)
        bonds_energy = self.bond_energy(radii_list)
        angles_energy = self.angle_energy(angles_list)
        dihedrals_energy = self.dihedral_energy(dihedrals_list)
        subtotal_energy = tf.reduce_sum(bonds_energy) + tf.reduce_sum(angles_energy) + tf.reduce_sum(dihedrals_energy)
        lj_forces = htf.compute_nlist_forces(mapped_nlist, lj_energy_total)
        other_forces = htf.compute_positions_forces(positions=positions, energy=subtotal_energy)
        total_energy = lj_energy + subtotal_energy
        # return lj_forces + other_forces, mapped_pos, total_energy, radii_list, angles_list, dihedrals_list
        # TODO: put back in the bonds angles dihedrals
        # TODO: see if we can plot loss over time, get final loss
        # TODO: update these outputs to spit out our trained parameters.
        return lj_forces + other_forces, positions, total_energy, self.lj_energy.w, self.bond_energy.w, self.angle_energy.w, self.dihedral_energy.w # lj_energy, bonds_energy, angles_energy, dihedrals_energy

nneighbor_cutoff = 32
model = TrajModel(nneighbor_cutoff=nneighbor_cutoff,
                 cg_num=12, # beads per molecule, not total
                 adjacency_matrix=adjacency_matrix,
                 CG_NN=nneighbor_cutoff,
                 cg_mapping=cg_mapping,
                 r_cut=set_rcut,
                 output_forces=False,
                 rcut=set_rcut,
                 check_nlist=False)

# all the 'None' here is so we only train on the energy
model.compile('Adam', ['MeanAbsoluteError', None, None, None, None, None])

system = simulate.System(molecule='PEEK', para_weight=1.0,
                         density=1.2, n_compounds=[100],
                         polymer_lengths=[4], forcefield='gaff',
                         assert_dihedrals=True, remove_hydrogens=True)

sim = simulate.Simulation(system, gsd_write=100, mode='cpu', dt=0.0001, r_cut=set_rcut, tf_model=model)

sim.quench(kT=1., n_steps=1e2, shrink_steps=2e2)

outputs = sim.tfcompute.outputs
cg_positions = outputs[0]
np.save('cg_positions.npy', cg_positions)
cg_energy = outputs[1]
np.save('cg_energy.npy', cg_energy)
lj_energy_params = outputs[2]
np.save('cg_lj_params.npy', lj_energy_params)
bond_energy_params = outputs[3]
np.save('cg_bond_params.npy', bond_energy_params)
angle_energy_params = outputs[4]
np.save('cg_angle_params.npy', angle_energy_params)
dihedral_energy_params = outputs[5]
np.save('cg_dihedral_params.npy', dihedral_energy_params)

