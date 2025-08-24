# BindNet

BindNet is a deep learning framework for predicting **protein–ligand binding affinities**.  
It leverages **SE(3)-equivariant transformers** to capture the 3D geometric and chemical features of molecular complexes, enabling more accurate and generalizable predictions compared to traditional GNNs.

## Key Features
- **SE(3) Transformer Backbone** – encodes rotationally and translationally equivariant 3D molecular structures.
- **Protein–Ligand Complex Representation** – processes atom-level interactions from refined structural datasets.
- **Binding Affinity Prediction** – trained to regress continuous affinity values for realistic docking and drug design tasks.
- **Modular Design** – built with PyTorch for easy extension and experimentation.

## Dataset
BindNet is trained and evaluated on the **PDBbind Refined Set**, a curated dataset of experimentally measured protein–ligand binding affinities with high-resolution structural data.

## Project Structure
BindNet/
├── data/          # dataset processing scripts
├── models/        # SE(3) transformer model definitions
├── training/      # training and evaluation scripts
├── utils/         # helper functions
└── README.md
