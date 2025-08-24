import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from rdkit import Chem
import pickle
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


def compute_spherical_harmonics(l_max, vectors):
    """Compute real spherical harmonics up to degree l_max for unit vectors"""
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    sh_features = {}

    # l = 0 (scalars) - normalized
    sh_features[0] = np.ones((len(vectors), 1)) * np.sqrt(1 / (4 * np.pi))

    # l = 1 (vectors)
    if l_max >= 1:
        sh_features[1] = np.sqrt(3 / (4 * np.pi)) * np.stack([
            -y,  # Y_1^{-1} (imaginary part)
            z,  # Y_1^0
            -x  # Y_1^1 (imaginary part)
        ], axis=1)

    # l = 2 (quadrupole) - normalized
    if l_max >= 2:
        sh_features[2] = np.stack([
            np.sqrt(15 / (4 * np.pi)) * x * y,  # Y_2^{-2}
            np.sqrt(15 / (4 * np.pi)) * y * z,  # Y_2^{-1}
            np.sqrt(5 / (16 * np.pi)) * (2 * z * z - x * x - y * y),  # Y_2^0
            np.sqrt(15 / (4 * np.pi)) * x * z,  # Y_2^1
            np.sqrt(15 / (16 * np.pi)) * (x * x - y * y)  # Y_2^2
        ], axis=1)

    return sh_features


def bessel_radial_basis(distances, num_basis=8, cutoff=10.0):
    """Bessel radial basis functions with normalization"""

    d_safe = np.maximum(distances, 1e-8)
    d_scaled = d_safe * np.pi / cutoff

    basis = np.zeros((len(distances), num_basis))
    for n in range(num_basis):
        basis[:, n] = np.sqrt(2.0 / cutoff) * np.sin((n + 1) * d_scaled) / d_scaled

    cutoff_mask = (distances < cutoff).astype(np.float32)
    return basis * cutoff_mask[:, None]


def polynomial_cutoff(distances, cutoff=10.0):
    """Polynomial cutoff function"""
    d_scaled = distances / cutoff
    mask = d_scaled < 1.0
    cutoff_vals = np.zeros_like(distances)

    # Polynomial: f(x) = 1 - 6x^5 + 15x^4 - 10x^3 for x in [0,1]
    x = d_scaled[mask]
    cutoff_vals[mask] = 1 - 6 * x ** 5 + 15 * x ** 4 - 10 * x ** 3

    return cutoff_vals


class GeometricProteinLigandEncoder:
    def __init__(self,
                 pocket_radius=8.0,
                 edge_cutoff=6.0,
                 l_max=2,
                 num_radial=8,
                 max_neighbors=20):
        self.pocket_radius = pocket_radius
        self.edge_cutoff = edge_cutoff
        self.l_max = l_max
        self.num_radial = num_radial
        self.max_neighbors = max_neighbors

        # atomic properties
        self.atom_props = {
            'C': [6, 2.55, 0.77, 4], 'N': [7, 3.04, 0.75, 3], 'O': [8, 3.44, 0.73, 2],
            'S': [16, 2.58, 1.02, 2], 'P': [15, 2.19, 1.06, 3], 'F': [9, 3.98, 0.71, 1],
            'Cl': [17, 3.16, 0.99, 1], 'Br': [35, 2.96, 1.14, 1], 'I': [53, 2.66, 1.33, 1],
            'H': [1, 2.20, 0.37, 1]
        }

        # amino acid properties (hydrophobicity, charge, volume, polarity)
        self.aa_props = {
            'ALA': [1.8, 0, 67, 0], 'ARG': [-4.5, 1, 148, 1], 'ASN': [-3.5, 0, 96, 1],
            'ASP': [-3.5, -1, 91, 1], 'CYS': [2.5, 0, 86, 0], 'GLU': [-3.5, -1, 109, 1],
            'GLN': [-3.5, 0, 114, 1], 'GLY': [-0.4, 0, 48, 0], 'HIS': [-3.2, 0, 118, 1],
            'ILE': [4.5, 0, 124, 0], 'LEU': [3.8, 0, 124, 0], 'LYS': [-3.9, 1, 135, 1],
            'MET': [1.9, 0, 124, 0], 'PHE': [2.8, 0, 135, 0], 'PRO': [-1.6, 0, 90, 0],
            'SER': [-0.8, 0, 73, 1], 'THR': [-0.7, 0, 93, 1], 'TRP': [-0.9, 0, 163, 0],
            'TYR': [-1.3, 0, 141, 1], 'VAL': [4.2, 0, 105, 0]
        }

        self.atom_encoder = LabelEncoder()
        self.residue_encoder = LabelEncoder()
        self.ligand_encoder = LabelEncoder()
        self.fitted = False

    def extract_pocket(self, protein_data, ligand_coords):
        """Extract pocket based on distance from any ligand atom"""
        protein_coords = np.array(protein_data['coordinates'])
        ligand_coords = np.array(ligand_coords)

        # Compute minimum distance from each protein atom to any ligand atom
        distances = cdist(protein_coords, ligand_coords)
        min_distances = np.min(distances, axis=1)

        # Select atoms within pocket radius
        pocket_mask = min_distances <= self.pocket_radius
        pocket_indices = np.where(pocket_mask)[0]

        if len(pocket_indices) == 0:
            # Fallback to closest atoms
            pocket_indices = np.argsort(min_distances)[:50]

        return {
            'coordinates': protein_coords[pocket_indices],
            'atom_names': [protein_data['atom_names'][i] for i in pocket_indices],
            'residue_names': [protein_data['residue_names'][i] for i in pocket_indices]
        }

    def fit(self, complexes_data):
        """Fit encoders on all data"""
        all_atoms, all_residues, all_ligand_atoms = set(), set(), set()

        for complex_data in complexes_data:
            ligand_coords = complex_data['ligand']['coordinates']
            pocket_data = self.extract_pocket(complex_data['protein'], ligand_coords)

            all_atoms.update(pocket_data['atom_names'])
            all_residues.update(pocket_data['residue_names'])
            all_ligand_atoms.update(complex_data['ligand']['atom_types'])

        self.atom_encoder.fit(list(all_atoms))
        self.residue_encoder.fit(list(all_residues))
        self.ligand_encoder.fit(list(all_ligand_atoms))
        self.fitted = True

    def create_geometric_node_features(self, coords, atom_names, residue_names, is_ligand=False):
        """Create geometrically meaningful node features"""
        n_atoms = len(coords)

        # Scalar features (invariant)
        if is_ligand:
            # Ligand features
            scalar_dim = 8
            scalar_features = np.zeros((n_atoms, scalar_dim))

            # Encoded atom type
            try:
                atom_indices = self.ligand_encoder.transform(atom_names)
                scalar_features[:, 0] = atom_indices / len(self.ligand_encoder.classes_)
            except:
                pass

            # Atomic properties
            for i, atom_name in enumerate(atom_names):
                atom_key = atom_name if atom_name in self.atom_props else 'C'
                scalar_features[i, 1:5] = self.atom_props[atom_key]

        else:
            # Protein features
            scalar_dim = 10
            scalar_features = np.zeros((n_atoms, scalar_dim))

            # Encoded features
            try:
                atom_indices = self.atom_encoder.transform(atom_names)
                residue_indices = self.residue_encoder.transform(residue_names)
                scalar_features[:, 0] = atom_indices / len(self.atom_encoder.classes_)
                scalar_features[:, 1] = residue_indices / len(self.residue_encoder.classes_)
            except:
                pass

            # Properties
            for i, (atom_name, residue_name) in enumerate(zip(atom_names, residue_names)):
                atom_key = atom_name[0] if len(atom_name) > 0 and atom_name[0] in self.atom_props else 'C'
                residue_key = residue_name if residue_name in self.aa_props else 'ALA'

                scalar_features[i, 2:6] = self.atom_props[atom_key]
                scalar_features[i, 6:10] = self.aa_props[residue_key]

        vector_features = np.zeros((n_atoms, 3, 3))  # 3 vector channels

        # Add some geometric initialization based on local environment
        if n_atoms > 1:
            # Compute center of mass
            com = np.mean(coords, axis=0)
            to_com = com - coords  # vectors pointing toward center
            to_com_norm = np.linalg.norm(to_com, axis=1, keepdims=True)
            to_com_norm = np.maximum(to_com_norm, 1e-8)
            to_com_unit = to_com / to_com_norm

            # First vector channel: direction to center of mass
            vector_features[:, 0, :] = to_com_unit

            # Add small random perturbation to break symmetry
            vector_features[:, 1:, :] = np.random.normal(0, 0.01, (n_atoms, 2, 3))

        # Tensor features (5D tensors that transform as l=2)
        tensor_features = np.zeros((n_atoms, 2, 5))  # 2 tensor channels

        # Initialize with small random values
        tensor_features = np.random.normal(0, 0.01, (n_atoms, 2, 5))

        return {
            'scalar': torch.from_numpy(scalar_features.astype(np.float32)),
            'vector': torch.from_numpy(vector_features.astype(np.float32)),
            'tensor': torch.from_numpy(tensor_features.astype(np.float32))
        }

    def create_geometric_edges(self, coords):
        """Create edges based on geometric criteria with neighbor limiting"""
        n_atoms = len(coords)
        distances = cdist(coords, coords)

        # Create edge list with distance cutoff
        edge_list = []

        for i in range(n_atoms):
            # Find neighbors within cutoff
            neighbors = np.where((distances[i] <= self.edge_cutoff) & (distances[i] > 0))[0]

            # Limit number of neighbors if too many
            if len(neighbors) > self.max_neighbors:
                neighbor_distances = distances[i, neighbors]
                closest_indices = np.argsort(neighbor_distances)[:self.max_neighbors]
                neighbors = neighbors[closest_indices]

            # Add edges
            for j in neighbors:
                edge_list.append([i, j])

        if len(edge_list) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        return np.array(edge_list).T

    def create_geometric_edge_features(self, coords, edge_indices):
        """Create geometrically meaningful edge features"""
        if edge_indices.shape[1] == 0:
            return self._empty_edge_features()

        row, col = edge_indices[0], edge_indices[1]

        # Edge vectors and distances
        edge_vectors = coords[col] - coords[row]  # r_j - r_i
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)

        # Normalize edge vectors (handle zero-length edges)
        edge_lengths_safe = np.maximum(edge_lengths, 1e-8)
        edge_unit_vectors = edge_vectors / edge_lengths_safe[:, None]

        # Spherical harmonics for angular information
        sh_features = compute_spherical_harmonics(self.l_max, edge_unit_vectors)

        # Radial basis functions for distance information
        radial_features = bessel_radial_basis(edge_lengths, self.num_radial, self.edge_cutoff)

        # Smooth cutoff
        cutoff_values = polynomial_cutoff(edge_lengths, self.edge_cutoff)

        return {
            'lengths': torch.from_numpy(edge_lengths.astype(np.float32)),
            'vectors': torch.from_numpy(edge_vectors.astype(np.float32)),
            'unit_vectors': torch.from_numpy(edge_unit_vectors.astype(np.float32)),
            'radial': torch.from_numpy(radial_features.astype(np.float32)),
            'cutoff': torch.from_numpy(cutoff_values.astype(np.float32)),
            'sh_0': torch.from_numpy(sh_features[0].astype(np.float32)),
            'sh_1': torch.from_numpy(sh_features[1].astype(np.float32)),
            'sh_2': torch.from_numpy(sh_features[2].astype(np.float32))
        }

    def _empty_edge_features(self):
        """Return empty edge features for graphs with no edges"""
        return {
            'lengths': torch.zeros(0, dtype=torch.float32),
            'vectors': torch.zeros(0, 3, dtype=torch.float32),
            'unit_vectors': torch.zeros(0, 3, dtype=torch.float32),
            'radial': torch.zeros(0, self.num_radial, dtype=torch.float32),
            'cutoff': torch.zeros(0, dtype=torch.float32),
            'sh_0': torch.zeros(0, 1, dtype=torch.float32),
            'sh_1': torch.zeros(0, 3, dtype=torch.float32),
            'sh_2': torch.zeros(0, 5, dtype=torch.float32)
        }

    def encode_complex(self, complex_data):
        """Encode a protein-ligand complex"""
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")

        # Extract pocket and ligand
        ligand_coords = np.array(complex_data['ligand']['coordinates'])
        pocket_data = self.extract_pocket(complex_data['protein'], ligand_coords)

        protein_coords = np.array(pocket_data['coordinates'])
        n_protein = len(protein_coords)
        n_ligand = len(ligand_coords)

        # Create node features
        protein_features = self.create_geometric_node_features(
            protein_coords,
            pocket_data['atom_names'],
            pocket_data['residue_names'],
            is_ligand=False
        )

        ligand_features = self.create_geometric_node_features(
            ligand_coords,
            complex_data['ligand']['atom_types'],
            ['LIG'] * n_ligand,  # Dummy residue names for ligand
            is_ligand=True
        )

        # Combine coordinates and features
        all_coords = np.vstack([protein_coords, ligand_coords])

        # Pad ligand scalar features to match protein dimension
        ligand_scalar_padded = torch.zeros(n_ligand, protein_features['scalar'].shape[1])
        ligand_scalar_padded[:, :ligand_features['scalar'].shape[1]] = ligand_features['scalar']

        combined_features = {
            'scalar': torch.cat([protein_features['scalar'], ligand_scalar_padded], dim=0),
            'vector': torch.cat([protein_features['vector'], ligand_features['vector']], dim=0),
            'tensor': torch.cat([protein_features['tensor'], ligand_features['tensor']], dim=0)
        }

        # Create edges
        edge_indices = self.create_geometric_edges(all_coords)
        edge_features = self.create_geometric_edge_features(all_coords, edge_indices)

        # Create masks and node types
        n_total = n_protein + n_ligand
        protein_mask = torch.zeros(n_total, dtype=torch.bool)
        protein_mask[:n_protein] = True
        ligand_mask = torch.zeros(n_total, dtype=torch.bool)
        ligand_mask[n_protein:] = True

        node_types = torch.cat([
            torch.zeros(n_protein, dtype=torch.long),  # 0 for protein
            torch.ones(n_ligand, dtype=torch.long)  # 1 for ligand
        ])

        return Data(
            # Geometric features
            x_scalar=combined_features['scalar'],  # Scalar features (l=0)
            x_vector=combined_features['vector'],  # Vector features (l=1)
            x_tensor=combined_features['tensor'],  # Tensor features (l=2)

            # Graph structure
            edge_index=torch.from_numpy(edge_indices.astype(np.int64)),
            pos=torch.from_numpy(all_coords.astype(np.float32)),

            # Edge features
            edge_lengths=edge_features['lengths'],
            edge_vectors=edge_features['vectors'],
            edge_unit_vectors=edge_features['unit_vectors'],
            edge_radial=edge_features['radial'],
            edge_cutoff=edge_features['cutoff'],
            edge_sh_0=edge_features['sh_0'],
            edge_sh_1=edge_features['sh_1'],
            edge_sh_2=edge_features['sh_2'],

            # Meta information
            node_type=node_types,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask,

            # Target
            y=torch.tensor([complex_data['affinity']], dtype=torch.float32),
            pdb_code=complex_data.get('pdb_code', 'unknown'),
            smiles=complex_data['ligand']['smiles']
        )

    def encode_dataset(self, complexes_data):
        """Encode a dataset of complexes"""
        encoded_data = []
        failed_count = 0

        for i, complex_data in enumerate(complexes_data):
            try:
                data = self.encode_complex(complex_data)
                encoded_data.append(data)

                if (i + 1) % 100 == 0:
                    print(f"Encoded {i + 1}/{len(complexes_data)} complexes")

            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"Failed to encode complex {i}: {str(e)}")
                continue

        print(f"Successfully encoded: {len(encoded_data)}, Failed: {failed_count}")
        return encoded_data


