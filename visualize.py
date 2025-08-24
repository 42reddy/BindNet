import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


class ComplexVisualizer:
    """
    Simple visualizer for protein-ligand complexes from processed PDBbind data
    """

    def __init__(self, pickle_file):
        """Load the processed data"""
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} complexes")

    def show_complex_info(self, idx=0):
        """Print basic info about a complex"""
        complex_data = self.data[idx]

        print(f"\n=== Complex {idx}: {complex_data['pdb_code']} ===")
        print(f"Binding Affinity (pKd): {complex_data['affinity']:.2f}")

        protein = complex_data['protein']
        ligand = complex_data['ligand']

        print(f"\nProtein:")
        print(f"  - {protein['num_atoms']} atoms")
        print(f"  - {protein['num_residues']} residues")
        print(f"  - Coordinate range: [{protein['coordinates'].min():.1f}, {protein['coordinates'].max():.1f}]")

        print(f"\nLigand:")
        print(f"  - SMILES: {ligand['smiles']}")
        print(f"  - {ligand['num_atoms']} atoms")
        print(f"  - Atom types: {set(ligand['atom_types'])}")
        print(f"  - Coordinate range: [{ligand['coordinates'].min():.1f}, {ligand['coordinates'].max():.1f}]")

    def visualize_complex_3d(self, idx=0, show_binding_site_only=True, binding_site_cutoff=8.0):
        """
        Create 3D visualization of protein-ligand complex

        Args:
            idx: Complex index to visualize
            show_binding_site_only: If True, only show protein atoms near ligand
            binding_site_cutoff: Distance cutoff for binding site (Angstroms)
        """
        complex_data = self.data[idx]
        protein = complex_data['protein']
        ligand = complex_data['ligand']

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get coordinates
        protein_coords = protein['coordinates']
        ligand_coords = ligand['coordinates']

        # Filter protein atoms to binding site if requested
        if show_binding_site_only:
            # Find protein atoms within cutoff of any ligand atom
            binding_site_mask = []
            for i, p_coord in enumerate(protein_coords):
                min_dist = np.min([np.linalg.norm(p_coord - l_coord)
                                   for l_coord in ligand_coords])
                binding_site_mask.append(min_dist <= binding_site_cutoff)

            binding_site_mask = np.array(binding_site_mask)
            protein_coords_plot = protein_coords[binding_site_mask]
            print(f"Showing {len(protein_coords_plot)} protein atoms within {binding_site_cutoff}Å of ligand")
        else:
            protein_coords_plot = protein_coords

        # Plot protein atoms (smaller, semi-transparent)
        ax.scatter(protein_coords_plot[:, 0],
                   protein_coords_plot[:, 1],
                   protein_coords_plot[:, 2],
                   c='lightblue', s=20, alpha=0.6, label=f'Protein ({len(protein_coords_plot)} atoms)')

        # Color ligand atoms by type
        atom_colors = {
            'C': 'gray',
            'N': 'blue',
            'O': 'red',
            'S': 'yellow',
            'P': 'orange',
            'F': 'green',
            'Cl': 'green',
            'Br': 'darkred',
            'I': 'purple'
        }

        # Plot ligand atoms (larger, colored by type)
        for atom_type in set(ligand['atom_types']):
            mask = np.array(ligand['atom_types']) == atom_type
            coords = ligand_coords[mask]
            color = atom_colors.get(atom_type, 'black')
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=color, s=100, alpha=0.8,
                       label=f'Ligand {atom_type} ({sum(mask)})')

        # Styling
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.legend()
        ax.set_title(f'Complex {complex_data["pdb_code"]} - pKd: {complex_data["affinity"]:.2f}\n'
                     f'Ligand: {ligand["smiles"][:50]}{"..." if len(ligand["smiles"]) > 50 else ""}')

        plt.tight_layout()
        plt.show()

    def plot_binding_site_distances(self, idx=0, max_distance=15.0):
        """Plot histogram of protein-ligand atom distances"""
        complex_data = self.data[idx]
        protein = complex_data['protein']
        ligand = complex_data['ligand']

        # Calculate all pairwise distances
        distances = []
        for p_coord in protein['coordinates']:
            for l_coord in ligand['coordinates']:
                dist = np.linalg.norm(p_coord - l_coord)
                if dist <= max_distance:  # Only include close contacts
                    distances.append(dist)

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=4.0, color='red', linestyle='--', label='Close contact (~4Å)')
        plt.axvline(x=8.0, color='orange', linestyle='--', label='Binding site (~8Å)')

        plt.xlabel('Distance (Å)')
        plt.ylabel('Number of atom pairs')
        plt.title(f'Protein-Ligand Distance Distribution - {complex_data["pdb_code"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Close contacts (<4Å): {sum(1 for d in distances if d < 4.0)}")
        print(f"Binding site contacts (<8Å): {sum(1 for d in distances if d < 8.0)}")

    def show_dataset_overview(self):
        """Show overview of entire dataset"""
        affinities = [d['affinity'] for d in self.data]
        protein_sizes = [d['protein']['num_residues'] for d in self.data]
        ligand_sizes = [d['ligand']['num_atoms'] for d in self.data]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Affinity distribution
        axes[0, 0].hist(affinities, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Binding Affinity (pKd)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Binding Affinities')
        axes[0, 0].axvline(np.mean(affinities), color='red', linestyle='--',
                           label=f'Mean: {np.mean(affinities):.2f}')
        axes[0, 0].legend()

        # Protein size distribution
        axes[0, 1].hist(protein_sizes, bins=20, alpha=0.7, edgecolor='black', color='lightblue')
        axes[0, 1].set_xlabel('Number of Residues')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Protein Size Distribution')

        # Ligand size distribution
        axes[1, 0].hist(ligand_sizes, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
        axes[1, 0].set_xlabel('Number of Atoms')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Ligand Size Distribution')

        # Affinity vs ligand size scatter
        axes[1, 1].scatter(ligand_sizes, affinities, alpha=0.6)
        axes[1, 1].set_xlabel('Ligand Size (atoms)')
        axes[1, 1].set_ylabel('Binding Affinity (pKd)')
        axes[1, 1].set_title('Affinity vs Ligand Size')

        plt.tight_layout()
        plt.show()

    def export_complex_for_modeling(self, idx=0):
        """Export a complex in a format ready for ML modeling"""
        complex_data = self.data[idx]

        # Create a simple feature representation
        features = {
            'pdb_code': complex_data['pdb_code'],
            'target': complex_data['affinity'],

            # Protein features
            'protein_coords': complex_data['protein']['coordinates'],
            'protein_atom_names': complex_data['protein']['atom_names'],
            'protein_residue_names': complex_data['protein']['residue_names'],

            # Ligand features
            'ligand_coords': complex_data['ligand']['coordinates'],
            'ligand_atom_types': complex_data['ligand']['atom_types'],
            'ligand_smiles': complex_data['ligand']['smiles'],

            # Simple molecular descriptors
            'num_protein_atoms': complex_data['protein']['num_atoms'],
            'num_ligand_atoms': complex_data['ligand']['num_atoms'],
            'num_residues': complex_data['protein']['num_residues']
        }

        return features


