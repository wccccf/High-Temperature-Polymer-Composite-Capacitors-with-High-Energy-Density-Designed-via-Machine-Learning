import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd


class ProcessedDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.data[self.keys[0]])

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.keys}


# load data
input_csv = "processed_tensors.csv"
output_npz = "processed_data.npz"
df = pd.read_csv(input_csv)
# Create a dictionary to store all column data
data_dict = {col: df[col].to_numpy() for col in df.columns}

# save to npz data
np.savez(output_npz, **data_dict)

num_samples = len(data_dict[next(iter(data_dict))])
indices = np.arange(num_samples)

train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)


def split_data(data_dict, indices):
    return {key: value[indices] for key, value in data_dict.items()}


# split data
train_data = split_data(data_dict, train_indices)
val_data = split_data(data_dict, val_indices)
test_data = split_data(data_dict, test_indices)

# create Dataset object
train_dataset = ProcessedDataset(train_data)
val_dataset = ProcessedDataset(val_data)
test_dataset = ProcessedDataset(test_data)

# create DataLoader
dataloaders = {
    'train': DataLoader(train_dataset, shuffle=True),
    'valid': DataLoader(val_dataset, shuffle=False),
    'test': DataLoader(test_dataset, shuffle=False),
}

# Extract data from the DataLoader and save it to a file
def save_dataloader_to_npz(dataloader, filename):
    dataset = dataloader.dataset
    data_dict = dataset.data
    np.savez(filename, **data_dict)


# save data
save_dataloader_to_npz(dataloaders['train'], "train_data.npz")
save_dataloader_to_npz(dataloaders['valid'], "valid_data.npz")
save_dataloader_to_npz(dataloaders['test'], "test_data.npz")
