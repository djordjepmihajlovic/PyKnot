import os
import torch
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

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
            data = np.loadtxt(os.path.join(dirname, fname))

        self.dataset = torch.tensor(data, dtype=torch.float32)

        # Reshape data
        self.dataset = self.dataset.view(-1, Nbeads, n_col)

        self.label = label

        if dtype == "XYZ":
            self.dataset = self.dataset - torch.mean(self.dataset, dim=0)

        if "CNN" in net:
            # self.dataset = self.dataset.unsqueeze(2)
            self.dataset = self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'label'):
            return self.dataset[idx], self.label
        else:
            return self.dataset[idx]
        
class CondKnotDataset(Dataset):

    def __init__(self, dirname, knot, net, dtype, Nbeads, pers_len, label):
        """Class wrapper for dataset generation --> conditional classification problem

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
        super(CondKnotDataset, self).__init__()

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
            data = np.loadtxt(os.path.join(dirname, fname))

        self.dataset = torch.tensor(data, dtype=torch.float32)

        # Reshape data
        self.dataset = self.dataset.view(-1, Nbeads, n_col)
        # reshape below is for the convolutional neural network, remember to change it back 
        # self.dataset = self.dataset.reshape(-1, 1, Nbeads, 3, 1)
        self.label = label
        self.conditional = torch.zeros(5)
        self.conditional[label] = 1

        if dtype == "XYZ":
            self.dataset = self.dataset - torch.mean(self.dataset, dim=0)

        if "CNN" in net:
            # self.dataset = self.dataset.unsqueeze(2)
            self.dataset = self.dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'label'):
            return self.dataset[idx], self.conditional, self.label
        else:
            return self.dataset[idx]
        
class data_2_inv(Dataset):

    def __init__(self, dirname, knot, net, dtype, Nbeads, pers_len, label, invariant):
        """Class wrapper for dataset generation --> knot invariant prediction problem

        Args:
            (example: SIGWRITHE --> DT code)
            dirname (str): knot master directory location
            knot (str): knot being called
            net (str): neural network trype
            dtype (str): data type used for features (eg. SIGWRITHE)
            Nbeads (int): number of beads
            pers_len (int): persistence length
            label: corresponding label of data being called
            invariant: invariant being predicted

        Returns:
            torch.Dataset
        """
        super(data_2_inv, self).__init__()

        header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)

        n_col_feature = len(select_cols)
        
        print((os.path.join(dirname, fname)))

        # Loading the dataset file

        if dtype == "SIGWRITHE":
            data = np.loadtxt(os.path.join(dirname,fname), usecols=(2,))

        if dtype == "2DSIGWRITHE":
            data = np.loadtxt(os.path.join(dirname, fname))

        self.knot = knot

        ## Loading the dataset labels, only needed for direct dowker prediction
        #on cluster use

        if invariant == "v2":
            labels = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/vassiliev/vassiliev_{self.knot}_v2_100,000.csv', delimiter=',', dtype=np.float32)
            # labels = np.loadtxt(f'/Users/s1910360/Desktop/ML for Knot Theory/sample_data/vassiliev/vassiliev_data/vassiliev_{knot}_v2_100,000.csv', delimiter=',', dtype=np.float32)

            self.label = torch.tensor(labels, dtype=torch.float32)
            self.label = self.label.view(-1, 1)

        if invariant == "v3":
            labels = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/vassiliev/vassiliev_data/vassiliev_{self.knot}_v3.csv', delimiter=',', dtype=np.float32)

            self.label = torch.tensor(labels, dtype=torch.float32)
            self.label = self.label.view(-1, 1)
        
        if invariant == "dowker":

            ## used for generated dowker code
            labels = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/knot data/dowker/dowker_{knot}_padded.csv', delimiter=',', dtype=np.float32)
            self.label = torch.tensor(labels, dtype=torch.float32)
            self.label = self.label.view(-1, 32, 1)


        self.dataset = torch.tensor(data, dtype=torch.float32)
        # self.dataset = self.dataset.view(-1, Nbeads, n_col_feature)

        if net == "CNN":
            self.dataset = self.dataset.view(-1, 1, Nbeads, n_col_feature)
        else:
            self.dataset = self.dataset.view(-1, Nbeads, n_col_feature)

        if dtype == "XYZ":
            self.dataset = self.dataset - torch.mean(self.dataset, dim=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'label'):
            per100 = math.floor(idx/100) # get label attributed to 100 bead section
            return self.dataset[idx], self.label[per100] 
        else:
            return self.dataset[idx]
        

class ConceptKnotDataset(Dataset):

    def __init__(self, dirname, knot, net, dtype, Nbeads, pers_len, label):
        """Class wrapper for dataset generation --> Concept bottleneck classifiers

        Args:
            (example: SIGWRITHE --> DT code)
            dirname (str): knot master directory location
            knot (str): knot being called
            net (str): neural network trype
            dtype (str): data type used for features (eg. SIGWRITHE)
            Nbeads (int): number of beads
            pers_len (int): persistence length
            concept: corresponding concept(s) of data being called
            label: corresponding label of data being called

        Returns:
            torch.Dataset
        """
        super(ConceptKnotDataset, self).__init__()

        header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)

        n_col = len(select_cols)
        
        print((os.path.join(dirname, fname)))

        self.label = label

        # Loading the dataset file

        if dtype == "SIGWRITHE":
            data = np.loadtxt(os.path.join(dirname,fname), usecols=(2,))

        if dtype == "2DSIGWRITHE":
            data = np.loadtxt(os.path.join(dirname, fname))

        self.knot = knot

        # Loading the dataset labels

        # concept1 = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/knot data/sta concepts/peaks prominence=0.75/peak count/peakcount_{knot}_prom=0.75.csv', delimiter=',', dtype=np.float32)
        # concept1 = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/knot data/sts concepts/peaks/peakcount_{knot}_5.csv', delimiter=',', dtype=np.float32)
        # concept2 = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/knot data/sta concepts/area/area_{knot}.csv', delimiter=',', dtype=np.float32)

        #concept1 = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/vassiliev/vassiliev_{knot}_v2_100,000.csv', delimiter=',', dtype=np.float32)
        #concept2 = np.loadtxt(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/vassiliev/vassiliev_{knot}_v3_10000.csv', delimiter=',', dtype=np.float32)

        # locally stored

        concept1 = np.loadtxt(f'/Users/s1910360/Desktop/ML for Knot Theory/sample_data/vassiliev/vassiliev_{knot}_v2_100,000.csv', delimiter=',', dtype=np.float32)
        concept2 = np.loadtxt(f'/Users/s1910360/Desktop/ML for Knot Theory/sample_data/vassiliev/vassiliev_{knot}_v3_10000.csv', delimiter=',', dtype=np.float32)

        self.concept1 = torch.tensor(concept1, dtype=torch.float32)
        self.concept1 = self.concept1.view(-1, 1, 1) 

        self.concept2 = torch.tensor(concept2, dtype=torch.float32)
        self.concept2 = self.concept2.view(-1, 1, 1)

        self.dataset = torch.tensor(data, dtype=torch.float32)
        self.dataset = self.dataset.view(-1, Nbeads, n_col)

        if dtype == "XYZ":
            self.dataset = self.dataset - torch.mean(self.dataset, dim=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lb = math.floor(idx/100) # get concepts attributed to 100 bead section
        return self.dataset[idx], self.concept1[lb], self.concept2[lb], self.label

        

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

    
