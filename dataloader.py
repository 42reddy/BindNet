import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
from rdkit import Chem
from biopandas.pdb import PandasPdb
import pickle
import warnings

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


class PDBBindProcessor:
    def __init__(self, refined_set_path="./refined-set"):
        self.refined_set_path = Path(refined_set_path)
        self.data = []

    def parse_affinity_value(self, affinity_str):
        """Parse affinity value from string format"""
        affinity_str = affinity_str.strip()
        pattern = r'([<>=]?)(\d+\.?\d*)'
        match = re.search(pattern, affinity_str)

        if not match:
            return None

        inequality, value_str = match.groups()

        try:
            value = float(value_str)
            return value
        except:
            return None

    def parse_index_file(self, index_file_path):
        """Parse the PDBbind index file to extract binding affinities"""
        affinities = {}

        print(f"Parsing index file: {index_file_path}")

        with open(index_file_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) >= 4:
                pdb_code = parts[0]
                affinity_str = parts[3]

                affinity = self.parse_affinity_value(affinity_str)

                if affinity is not None:
                    affinities[pdb_code] = affinity
                else:
                    if len(affinities) < 10:  # debug output
                        print(f"Could not parse affinity: {pdb_code} -> {affinity_str}")

        print(f"Found {len(affinities)} complexes with valid binding data")

        if affinities:
            values = list(affinities.values())
            print(f"Affinity range: {min(values):.2f} - {max(values):.2f} pK units")
            print(f"Mean affinity: {np.mean(values):.2f} ± {np.std(values):.2f}")

        return affinities

    def get_complex_files(self, pdb_code):
        """Find protein and ligand files for a given PDB code"""
        complex_dir = self.refined_set_path / pdb_code

        if not complex_dir.exists():
            return None, None

        protein_file = None
        ligand_file = None

        for file_path in complex_dir.iterdir():
            filename = file_path.name.lower()

            # protein files
            if filename.endswith('_protein.pdb') or (filename.endswith('.pdb') and 'protein' in filename):
                protein_file = file_path
            elif filename == f"{pdb_code}_protein.pdb":
                protein_file = file_path
            elif file_path.suffix == '.pdb' and protein_file is None:
                protein_file = file_path

            # ligand files
            if filename.endswith('_ligand.mol2') or filename.endswith('_ligand.sdf'):
                ligand_file = file_path
            elif filename == f"{pdb_code}_ligand.mol2" or filename == f"{pdb_code}_ligand.sdf":
                ligand_file = file_path
            elif file_path.suffix in ['.mol2', '.sdf'] and ligand_file is None:
                ligand_file = file_path

        return protein_file, ligand_file

    def process_protein(self, pdb_file):
        """Extract protein structural information"""
        try:
            ppdb = PandasPdb().read_pdb(str(pdb_file))
            atoms = ppdb.df['ATOM']

            if len(atoms) == 0:
                return None

            return {
                'num_atoms': len(atoms),
                'num_residues': len(atoms['residue_number'].unique()),
                'coordinates': atoms[['x_coord', 'y_coord', 'z_coord']].values,
                'atom_names': atoms['atom_name'].values.tolist(),
                'residue_names': atoms['residue_name'].values.tolist()
            }
        except Exception as e:
            print(f"Error processing protein {pdb_file}: {e}")
            return None

    def process_ligand(self, ligand_file):
        """Extract ligand structural information"""
        try:
            mol = None

            if ligand_file.suffix == '.mol2':
                mol = Chem.MolFromMol2File(str(ligand_file))
            elif ligand_file.suffix == '.sdf':
                mol = Chem.MolFromMolFile(str(ligand_file))

            if mol is None:
                return None

            conf = mol.GetConformer()
            coords = []
            atom_types = []

            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
                atom_types.append(mol.GetAtomWithIdx(i).GetSymbol())

            return {
                'smiles': Chem.MolToSmiles(mol),
                'num_atoms': mol.GetNumAtoms(),
                'coordinates': np.array(coords),
                'atom_types': atom_types
            }
        except Exception as e:
            print(f"Error processing ligand {ligand_file}: {e}")
            return None

    def process_complexes(self, index_file_path, max_complexes=None):
        """Process protein-ligand complexes from PDBbind dataset"""
        affinities = self.parse_index_file(index_file_path)

        if not affinities:
            print("No valid affinities found!")
            return

        processed = 0
        failed = 0

        sorted_complexes = sorted(affinities.items())

        for pdb_code, affinity in sorted_complexes:
            if max_complexes and processed >= max_complexes:
                break

            protein_file, ligand_file = self.get_complex_files(pdb_code)

            if protein_file is None or ligand_file is None:
                failed += 1
                continue

            protein_info = self.process_protein(protein_file)
            ligand_info = self.process_ligand(ligand_file)

            if protein_info is None or ligand_info is None:
                failed += 1
                continue

            self.data.append({
                'pdb_code': pdb_code,
                'affinity': affinity,
                'protein': protein_info,
                'ligand': ligand_info
            })

            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} complexes...")

        print(f"Successfully processed: {processed}")
        print(f"Failed: {failed}")

    def save_data(self, output_file="processed_pdbbind.pkl"):
        """Save processed data to pickle file"""
        with open(output_file, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Saved {len(self.data)} complexes to {output_file}")

    def load_data(self, input_file):
        """Load processed data from pickle file"""
        with open(input_file, 'rb') as f:
            self.data = pickle.load(f)

        print(f"Loaded {len(self.data)} complexes")
        return self.data

    def get_stats(self):
        """Print dataset statistics"""
        if not self.data:
            print("No data processed")
            return

        affinities = [d['affinity'] for d in self.data]
        protein_sizes = [d['protein']['num_residues'] for d in self.data]
        ligand_sizes = [d['ligand']['num_atoms'] for d in self.data]

        print(f"\nDataset Statistics:")
        print(f"Number of complexes: {len(self.data)}")
        print(f"Affinity (pK units): {np.mean(affinities):.2f} ± {np.std(affinities):.2f}")
        print(f"Affinity range: {min(affinities):.2f} - {max(affinities):.2f}")
        print(f"Protein size: {np.mean(protein_sizes):.0f} ± {np.std(protein_sizes):.0f} residues")
        print(f"Ligand size: {np.mean(ligand_sizes):.1f} ± {np.std(ligand_sizes):.1f} atoms")

        # affinity distribution
        hist, bins = np.histogram(affinities, bins=20)
        print(f"\nAffinity distribution:")
        for i in range(len(hist)):
            if hist[i] > 0:
                print(f"  {bins[i]:.2f}-{bins[i + 1]:.2f}: {hist[i]} complexes")

    def get_training_data(self):
        """Get data formatted for model training"""
        if not self.data:
            return None, None, None

        features = []
        targets = []
        pdb_codes = []

        for complex_data in self.data:
            features.append({
                'protein': complex_data['protein'],
                'ligand': complex_data['ligand']
            })
            targets.append(complex_data['affinity'])
            pdb_codes.append(complex_data['pdb_code'])

        return features, np.array(targets), pdb_codes



