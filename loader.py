import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import functional as F

from helper import datafile_structure


class Wr_2_XYZ(Dataset):
    def __init__(self, dirname, knot, net, dtype_f, dtype_l, Nbeads, pers_len, label):
        super(Wr_2_XYZ, self).__init__()

        header, fname_f, select_cols_f = datafile_structure(dtype_f, knot, Nbeads, pers_len)
        header, fname_l, select_cols_l = datafile_structure(dtype_l, knot, Nbeads, pers_len)
        # select_cols = [0, 1, 2] for XYZ [2] for SIGWRITHE

        n_col_feature = len(select_cols_f)
        n_col_label = len(select_cols_l)
        
        print((os.path.join(dirname, fname_f)))
        print((os.path.join(dirname, fname_l)))
        # Loading the dataset file
        data = np.loadtxt(os.path.join(dirname,fname_f), usecols=(2,))
        label = np.loadtxt(os.path.join(dirname,fname_l))

        self.dataset = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

        # Reshape data
        self.dataset = self.dataset.view(-1, Nbeads, n_col_feature)
        self.label = self.label.view(-1, Nbeads, n_col_label)

        if dtype_l == "XYZ":
            self.label = self.label - torch.mean(self.label, dim=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'label'):
            return self.dataset[idx], self.label[idx]
        else:
            return self.dataset[idx]
        

class KnotDataset(Dataset):
    def __init__(self, dirname, knot, net, dtype, Nbeads, pers_len, label):
        super(KnotDataset, self).__init__()

        header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)

        #fname = (f"XYZ_{knot}.dat.nos")
        #select_cols = [0, 1, 2] for XYZ [2] for SIGWRITHE

        n_col = len(select_cols)
        type_list = [torch.float32] * n_col
        
        print((os.path.join(dirname, fname)))

        # Loading the dataset file

        if dtype == "XYZ":
            data = np.loadtxt(os.path.join(dirname, fname))

        if dtype == "SIGWRITHE":
            data = np.loadtxt(os.path.join(dirname,fname), usecols=(2,))

        self.dataset = torch.tensor(data, dtype=torch.float32)

        # Reshape data
        self.dataset = self.dataset.view(-1, Nbeads, n_col)
        self.label = label

        if dtype == "XYZ":
            self.dataset = self.dataset - torch.mean(self.dataset, dim=0)

        if "CNN" in net:
            self.dataset = self.dataset.unsqueeze(2)

        # Add Kymoknot labels if loading for a localization problem
        if "localise" in net:
            label_data = np.loadtxt(dirname + f"KYMOKNOT/BU__KN_{knot}.dat.cleaned")[:, 2]
            label_dataset = torch.tensor(label_data, dtype=torch.float32)
            label_dataset = label_dataset.view(-1, Nbeads, 1)
            self.dataset = TensorDataset(self.dataset, label_dataset)

        elif "FOR" in net:
            self.dataset = self.dataset.view(-1, Nbeads * n_col)
            self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'label'):
            return self.dataset[idx], self.label
        else:
            return self.dataset[idx]

def split_train_test_validation(dataset, train_size, test_size, val_size, batch_size):
    """Generate splitted dataset

    Args:
        dataset (Dataset): Total dataset
        train_size (int): size of the train dataset
        test_size (int): size of the test dataset
        val_size (int): size of the validation dataset

    Returns:
        DataLoader: train dataset
        DataLoader: test dataset
        DataLoader: validation dataset
    """
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size], generator = generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader

def pad_sequences(x):
    """Padding sequences to 1000

    Args:
        x (torch.Tensor): input sequence

    Returns:
        torch.Tensor: padded sequence
    """
    x = F.pad(x, (0, 1000 - x.size(1)), value=-100)
    return x


    
