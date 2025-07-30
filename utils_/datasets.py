import os
import os.path as osp
import pickle
import random

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from sklearn.utils import shuffle
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.data import InMemoryDataset, download_url
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')



class CustomDataset(InMemoryDataset):
    def __init__(self, split, root='test', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.folder = osp.join(root, 'custom')

        super(CustomDataset, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.processed_paths[0] = osp.join(root, 'custom_processed.pt')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f"{self.split}.npz"  # Your raw data file name, e.g., 'train.npz'

    @property
    def processed_file_names(self):
        return f"custom_{self.split}.pt"  # Processed data file name

    def process(self):
        raw_path = osp.normpath(osp.join('test', self.raw_file_names))
        data_npz = np.load(raw_path)

        # Extract data from npz file
        positions = data_npz['positions']  # Shape: [num_graphs, max_atoms, 3]
        charges = data_npz['charges']  # Shape: [num_graphs, max_atoms]
        num_atoms = data_npz['num_atoms']  # Shape: [num_graphs]
        homo = data_npz['homo']  # Shape: [num_graphs]
        lumo = data_npz['lumo']  # Shape: [num_graphs]

        data_list = []
        for i in tqdm(range(len(num_atoms))):
            # Process each graph
            num_atoms_i = num_atoms[i]
            pos_i = torch.tensor(positions[i][:num_atoms_i,:], dtype=torch.float32)
            z_i = torch.tensor(charges[i][:num_atoms_i], dtype=torch.int64)
            homo_i = torch.tensor(homo[i], dtype=torch.float32)
            lumo_i = torch.tensor(lumo[i], dtype=torch.float32)
            # row_id_i = torch.tensor(row_id[i], dtype=torch.long)


            # Create the Data object
            data = Data(
                pos=pos_i,
                atom_type=z_i,
                homo=homo_i,
                lumo=lumo_i,
                # y=torch.stack([homo_i, lumo_i], dim=0), # Regression targets
                num_atoms=torch.tensor(num_atoms_i)
            )

            data_list.append(data)

        # Apply optional pre-filtering and pre-transforming
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving processed data...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
