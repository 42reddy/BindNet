from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from dataloader import PDBBindProcessor
from visualize import ComplexVisualizer
import pickle
from preprocessing import GeometricProteinLigandEncoder
from augment import ProteinLigandAugmentor
from GAT import SE3AllFeatures
from tqdm import tqdm


processor = PDBBindProcessor("./refined-set")

processor.process_complexes("./refined-set/index/INDEX_general_PL_data.2020", max_complexes=5000)
processor.get_stats()
processor.save_data("pdbbind_5k.pkl")


viz = ComplexVisualizer("pdbbind_5k.pkl")

# Show info about first complex
viz.show_complex_info(20)

# Create 3D visualization
viz.visualize_complex_3d(20, show_binding_site_only=True)

# Show distance distribution
viz.plot_binding_site_distances(20)



# Load data
with open("pdbbind_5k.pkl", "rb") as f:
    complexes_data = pickle.load(f)

# Create encoder with geometric focus
encoder = GeometricProteinLigandEncoder(
    pocket_radius=8.0,  # Larger pocket for more context
    edge_cutoff=6.0,  # Reasonable cutoff for chemical interactions
    l_max=2,  # Up to quadrupole moments
    num_radial=8,  # Sufficient radial basis functions
    max_neighbors=15  # Limit connectivity for efficiency
)

# Fit and encode
encoder.fit(complexes_data)
encoded_data = encoder.encode_dataset(complexes_data)

# Print statistics
if encoded_data:
    data = encoded_data[0]

# Save encoded data
#torch.save(encoded_data, "geometric_se3_complexes.pt")

dataset = encoded_data

n_samples = len(dataset)
indices = list(range(n_samples))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = [dataset[i] for i in train_idx]
val_dataset = [dataset[i] for i in val_idx]
test_dataset = [dataset[i] for i in test_idx]

val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

augmentor = ProteinLigandAugmentor()
augmented_train_dataset = augmentor.augment_dataset(train_dataset, augment_ratio=10.0)

train_loader = DataLoader(augmented_train_dataset, batch_size=32, shuffle=Trule, pin_memory = True)





# initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SE3AllFeatures(
    num_node_scalar=10,
    num_node_type=200,        # set to (max node_type in your dataset) + 1; use a safe upper bound if unknown
    node_type_emb_dim=8,
    hidden_irreps="16x0e + 8x1o + 4x2e",
    readout_scalar_dim=32
).to(device)

print("Model initialized. Parameters:", sum(p.numel() for p in model.parameters()))



# trainign setup

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay = 2e-4)
criterion = nn.MSELoss()

def train():
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(batch)#.squeeze(1)
        loss = criterion(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)#.squeeze(1)
            loss = criterion(out, batch.y.float())
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

for epoch in range(25):
    train_loss = train()
    val_loss = test(val_loader)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

test_loss = test(test_loader)

