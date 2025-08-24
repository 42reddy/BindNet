import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, Linear


def concat_sh(sh0, sh1, sh2):
    return torch.cat([sh0, sh1, sh2], dim=-1)

def masked_mean(x, mask, batch):
    mask = mask.float().unsqueeze(-1)
    num_graphs = int(batch.max().item()) + 1
    x_sum = scatter_add(x * mask, batch, dim=0, dim_size=num_graphs)
    cnt = scatter_add(mask, batch, dim=0, dim_size=num_graphs).clamp_min(1.0)
    return x_sum / cnt


class SE3MP(nn.Module):
    def __init__(self, irreps_in: Irreps, irreps_out: Irreps, sh_irreps: Irreps, edge_feat_dim: int):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.sh_irreps = Irreps(sh_irreps)

        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.sh_irreps, self.irreps_out, shared_weights=False)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel)
        )

        self.skip = Linear(self.irreps_in, self.irreps_out)

    def forward(self, node_feats, edge_index, edge_feat, edge_sh):
        src, dst = edge_index

        w = self.edge_mlp(edge_feat)
        m_ij = self.tp(node_feats[src], edge_sh, w)

        agg = torch.zeros(node_feats.size(0), self.irreps_out.dim, device=node_feats.device)
        agg.index_add_(0, dst, m_ij)

        skip = self.skip(node_feats)
        return skip + agg


class SE3AllFeatures(nn.Module):
    def __init__(
        self,
        num_node_scalar=10,
        num_node_type=32,
        node_type_emb_dim=8,
        hidden_irreps="16x0e + 8x1o + 4x2e",
        readout_scalar_dim=32
    ):
        super().__init__()

        # input irreps: scalars + vectors + tensors
        in_scalars = num_node_scalar + node_type_emb_dim + 2  # masks
        num_vector_copies = 3  # x_vector shape
        num_l2_copies = 2  # x_tensor shape

        self.irreps_in = Irreps(f"{in_scalars}x0e + {num_vector_copies}x1o + {num_l2_copies}x2e")
        self.irreps_hidden = Irreps(hidden_irreps)
        self.sh_irreps = Irreps("1x0e + 1x1o + 1x2e")

        self.node_type_emb = nn.Embedding(num_node_type, node_type_emb_dim)
        self.in_lin = Linear(self.irreps_in, self.irreps_hidden)

        self.edge_feat_dim = 8 + 1 + 1  # radial + cutoff + lengths

        self.mp1 = SE3MP(self.irreps_hidden, self.irreps_hidden, self.sh_irreps, edge_feat_dim=self.edge_feat_dim)
        self.mp2 = SE3MP(self.irreps_hidden, self.irreps_hidden, self.sh_irreps, edge_feat_dim=self.edge_feat_dim)

        self.scalar_norm = nn.LayerNorm(self.irreps_hidden.dim)
        self.readout_proj = Linear(self.irreps_hidden, f"{readout_scalar_dim}x0e")

        self.head = nn.Sequential(
            nn.Linear(readout_scalar_dim * 3, 64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

    def _pack_input(self, data):
        """Pack input features according to irreps ordering"""
        x_scalar = data.x_scalar
        node_emb = self.node_type_emb(data.node_type)
        masks = torch.cat([data.protein_mask.unsqueeze(-1).float(), data.ligand_mask.unsqueeze(-1).float()], dim=-1)

        scalars = torch.cat([x_scalar, node_emb, masks], dim=-1)
        vectors = data.x_vector.reshape(data.x_vector.shape[0], -1)
        tensors = data.x_tensor.reshape(data.x_tensor.shape[0], -1)

        packed = torch.cat([scalars, vectors, tensors], dim=-1)
        return packed

    def forward(self, data):
        x_in = self._pack_input(data)
        h = self.in_lin(x_in)

        # edge features
        edge_radial = data.edge_radial
        edge_cutoff = data.edge_cutoff.unsqueeze(-1)
        edge_len = data.edge_lengths.unsqueeze(-1)
        edge_feat = torch.cat([edge_radial, edge_cutoff, edge_len], dim=-1)

        edge_sh = concat_sh(data.edge_sh_0, data.edge_sh_1, data.edge_sh_2)

        # message passing
        h = self.mp1(h, data.edge_index, edge_feat, edge_sh)
        h = F.silu(h)
        h = self.scalar_norm(h)

        h = self.mp2(h, data.edge_index, edge_feat, edge_sh)
        h = F.silu(h)
        h = self.scalar_norm(h)

        # readout
        h_scalar = self.readout_proj(h)

        prot_mean = masked_mean(h_scalar, data.protein_mask, data.batch)
        lig_mean = masked_mean(h_scalar, data.ligand_mask, data.batch)
        all_mean = global_mean_pool(h_scalar, data.batch)

        graph_feat = torch.cat([prot_mean, lig_mean, all_mean], dim=-1)
        out = self.head(graph_feat).squeeze(-1)

        return out