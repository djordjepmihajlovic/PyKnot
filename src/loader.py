import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import functional as F

from helper import datafile_structure

class KnotDataset(Dataset):

    def __init__(self, dirname, knot, net, dtype, Nbeads, pers_len, label):
        """Class wrapper for dataset generation --> classification problem

        Args:
            dirname (str): knot master directory location
            knot (str): knot being called
            net (str): neural network trype
            dtype (str): problem type
            Nbeads (int): number of beads
            pers_len (int): persistence length
            label (int): corresponding label of data being called

        Returns:
            torch.Dataset
        """
        super(KnotDataset, self).__init__()

        header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)

        n_col = len(select_cols)
        type_list = [torch.float32] * n_col
        
        print((os.path.join(dirname, fname)))

        # Loading the dataset file

        if dtype == "XYZ":
            data = np.loadtxt(os.path.join(dirname, fname))

        if dtype == "SIGWRITHE":
            data = np.loadtxt(os.path.join(dirname,fname), usecols=(2,))

        if dtype == "2DSIGWRITHE":
            data_attention = np.loadtxt(os.path.join(dirname, fname))

        self.dataset = torch.tensor(data, dtype=torch.float32)

        # Reshape data
        self.dataset = self.dataset.view(-1, Nbeads, n_col)
        self.label = label

        if dtype == "XYZ":
            self.dataset = self.dataset - torch.mean(self.dataset, dim=0)

        if "CNN" in net:
            # self.dataset = self.dataset.unsqueeze(2)
            self.dataset = self.dataset

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
        
class StA_2_DT(Dataset):

    def __init__(self, dirname, knot, net, dtype, Nbeads, pers_len, label):
        """Class wrapper for dataset generation --> prediction problem

        Args:
            (example: SIGWRITHE --> DT code prediction)
            dirname (str): knot master directory location
            knot (str): knot being called
            net (str): neural network trype
            dtype (str): data type used for features (eg. SIGWRITHE)
            Nbeads (int): number of beads
            pers_len (int): persistence length
            label: corresponding label of data being called

        Returns:
            torch.Dataset
        """
        super(StA_2_DT, self).__init__()

        header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)

        # self.dowker_codes = {0:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        #                  1:[4.0, 6.0, 2.0, 0.0, 0.0, 0.0, 0.0], 
        #                  2:[4.0, 6.0, 8.0, 2.0, 0.0, 0.0, 0.0], 
        #                  3:[6.0, 8.0, 10.0, 2.0, 4.0, 0.0, 0.0], 
        #                  4:[4.0, 8.0, 10.0, 2.0, 6.0, 0.0, 0.0], 
        #                  5:[4.0, 8.0, 12.0, 10.0, 2.0, 6.0, 0.0], 
        #                  6:[4.0, 8.0, 10.0, 12.0, 2.0, 6.0, 0.0], 
        #                  7:[4.0, 8.0, 10.0, 2.0, 12.0, 6.0, 0.0], 
        #                  8:[8.0, 10.0, 12.0, 14.0, 2.0, 4.0, 6.0], 
        #                  9:[4.0, 10.0, 14.0, 12.0, 2.0, 8.0, 6.0], 
        #                  10:[6.0, 10.0, 12.0, 14.0, 2.0, 4.0, 8.0]} # dowker-code 
        self.dowker_codes = {0:[10.0, -6.0, -8.0, -4.0, 12.0, 2.0, 0.0, 0.0], # sq
                             1:[10.0, 6.0, 8.0, 4.0, 12.0, 2.0, 0.0, 0.0], # gr
                             2:[12.0, -8.0, 16.0, -2.0, -14.0, -4.0, 6.0, -10.0]} # 8_20
        n_col_feature = len(select_cols)
        
        print((os.path.join(dirname, fname)))
        # Loading the dataset file
        data = np.loadtxt(os.path.join(dirname,fname), usecols=(2,))
        self.knot = label
        self.label = torch.tensor(self.dowker_codes[self.knot])

        self.dataset = torch.tensor(data, dtype=torch.float32)

        # Reshape data
        self.dataset = self.dataset.view(-1, Nbeads, n_col_feature)
        self.label = self.label.view(8, 1)

        if dtype == "XYZ":
            self.dataset = self.dataset - torch.mean(self.dataset, dim=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'label'):
            return self.dataset[idx], self.label
        else:
            return self.dataset[idx]
        
class WeightedKnotDataset(Dataset):  # right now going to set up so that it takes in StS and creates XYZ

    def __init__(self, dirname, knot, net, dtype, Nbeads, pers_len, label):
        """Class wrapper for dataset generation --> classification problem

        Args:
            dirname (str): knot master directory location
            knot (str): knot being called
            net (str): neural network trype
            dtype (str): problem type
            Nbeads (int): number of beads
            pers_len (int): persistence length
            label (int): corresponding label of data being called

        Returns:
            three outputs: 
            torch.Tensor()
                dataset[idx], weights (StS)[idx], label
        """
        super(WeightedKnotDataset, self).__init__()

        header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)
        fname_w = os.path.join("SIGWRITHEMATRIX", f"3DSignedWritheMatrix_{knot}.dat.lp{pers_len}.dat")

        n_col = len(select_cols)
        type_list = [torch.float32] * n_col

        # Loading the dataset file

        if dtype == "XYZ":
            data_label = np.loadtxt(os.path.join(dirname, fname))

        if dtype == "SIGWRITHE":
            data = np.loadtxt(os.path.join(dirname,fname), usecols=(2,))

        print((os.path.join(dirname, fname)))

        data_feature = np.loadtxt(os.path.join(dirname, fname_w))
        print((os.path.join(dirname, fname_w)))

        self.dataset_label = torch.tensor(data_label, dtype=torch.float32)
        self.dataset_feature = torch.tensor(data_feature, dtype=torch.float32)

        # Reshape data
        self.dataset_label = self.dataset_label.view(-1, Nbeads, n_col) #XYZ
        self.dataset_feature = self.dataset_feature.view(-1, Nbeads, len(np.arange(Nbeads))) #StS

        self.label = label

        if dtype == "XYZ":
            self.dataset_label = self.dataset_label - torch.mean(self.dataset_label, dim=0)

        # Add Kymoknot labels if loading for a localization problem
        if "localise" in net:
            label_data = np.loadtxt(dirname + f"KYMOKNOT/BU__KN_{knot}.dat.cleaned")[:, 2]
            label_dataset = torch.tensor(label_data, dtype=torch.float32)
            label_dataset = label_dataset.view(-1, Nbeads, 1)
            self.dataset = TensorDataset(self.dataset, label_dataset)


    def __len__(self):
        return len(self.dataset_feature)

    def __getitem__(self, idx):
        if hasattr(self, 'label'):
            return self.dataset_feature[idx], self.dataset_label[idx], self.label
        else:
            return self.dataset[idx]
        

        

def split_train_test_validation(dataset, train_size, test_size, val_size, batch_size):
    """Generate split dataset
    
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
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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


    
