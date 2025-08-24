import torch
import copy
import numpy as np
import random


class ProteinLigandAugmentor:
    def __init__(self):
        self.translation_std = 0.3
        self.noise_std = 0.015
        self.scalar_noise_std = 0.02
        self.dropout_prob = 0.03
        self.ligand_perturb_std = 0.08

        self.translation_prob = 0.5
        self.noise_prob = 0.6
        self.scalar_noise_prob = 0.4
        self.dropout_prob_apply = 0.2
        self.ligand_perturb_prob = 0.3

    def random_rotation_matrix(self):
        alpha = np.random.uniform(0, 2 * np.pi)
        beta = np.random.uniform(0, 2 * np.pi)
        gamma = np.random.uniform(0, 2 * np.pi)

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])

        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx
        return torch.tensor(R, dtype=torch.float32)

    def rotate_data(self, data):
        R = self.random_rotation_matrix().to(data.pos.device)
        new_data = copy.deepcopy(data)
        new_data.pos = data.pos @ R.T

        if hasattr(data, 'x_vector'):
            new_data.x_vector = torch.einsum('ij,njk->nik', R, data.x_vector)
        if hasattr(data, 'edge_vectors'):
            new_data.edge_vectors = data.edge_vectors @ R.T
        if hasattr(data, 'edge_unit_vectors'):
            new_data.edge_unit_vectors = data.edge_unit_vectors @ R.T

        return new_data

    def translate_data(self, data):
        if torch.rand(1) < self.translation_prob:
            new_data = copy.deepcopy(data)
            translation = torch.randn(3, device=data.pos.device) * self.translation_std
            new_data.pos = data.pos + translation
            return new_data
        return data

    def add_noise(self, data):
        if torch.rand(1) < self.noise_prob:
            new_data = copy.deepcopy(data)
            noise = torch.randn_like(data.pos) * self.noise_std
            new_data.pos = data.pos + noise
            return new_data
        return data

    def scalar_noise(self, data):
        if torch.rand(1) < self.scalar_noise_prob and hasattr(data, 'x_scalar'):
            new_data = copy.deepcopy(data)
            noise = torch.randn_like(data.x_scalar) * self.scalar_noise_std
            new_data.x_scalar = data.x_scalar + noise
            return new_data
        return data

    def node_dropout(self, data):
        if torch.rand(1) < self.dropout_prob_apply:
            new_data = copy.deepcopy(data)
            num_nodes = data.pos.size(0)
            keep_mask = torch.rand(num_nodes, device=data.pos.device) > self.dropout_prob

            if hasattr(data, 'x_scalar'):
                new_data.x_scalar = data.x_scalar * keep_mask.unsqueeze(-1).float()
            if hasattr(data, 'x_vector'):
                new_data.x_vector = data.x_vector * keep_mask.unsqueeze(-1).unsqueeze(-1).float()
            return new_data
        return data

    def ligand_perturbation(self, data):
        if torch.rand(1) < self.ligand_perturb_prob and hasattr(data, 'ligand_mask'):
            new_data = copy.deepcopy(data)
            ligand_indices = data.ligand_mask.nonzero().squeeze(-1)
            if len(ligand_indices) > 0:
                perturbation = torch.randn(len(ligand_indices), 3, device=data.pos.device) * self.ligand_perturb_std
                new_data.pos[ligand_indices] += perturbation
            return new_data
        return data

    def augment_single(self, data):
        new_data = self.rotate_data(data)
        new_data = self.translate_data(new_data)
        new_data = self.add_noise(new_data)
        new_data = self.scalar_noise(new_data)
        new_data = self.node_dropout(new_data)
        new_data = self.ligand_perturbation(new_data)
        return new_data

    def augment_dataset(self, dataset, augment_ratio=2.0):
        augmented = list(dataset)
        num_to_augment = int(len(dataset) * augment_ratio)

        for i in range(num_to_augment):
            original_idx = i % len(dataset)
            original_data = dataset[original_idx]
            augmented_data = self.augment_single(original_data)
            augmented.append(augmented_data)

        return augmented

