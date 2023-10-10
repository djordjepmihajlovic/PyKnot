import pandas as pd
import os
import torch 
from torch.utils.data import Dataset, DataLoader, random_split

from helper import datafile_structure

file_path = "XYZ_0_1.dat.nos"
dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/XYZ"
Nbeads = 100
dtype = "XYZ"
n_col = len([0, 1, 2])

class KnotDataset(Dataset):
    
    # reading the csv and defining predictor and output columns
    def __init__(self, dirname, file_path):

        # store the input and output features
        self.df = pd.read_csv(os.path.join(dirname, file_path), delimiter = " ", header = None)
        self.dataset = torch.tensor(self.df.to_numpy()).float()
    
    # number of rows in dataset
    def __len__(self):
        return len(self.dataset)
    
    # get a row at an index
    def __getitem__(self, index):
        return self.dataset[index]
    
def load_dataset(dirname, knot, net, dtype, Nbeads, pers_len, label):

    header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)
    
    knot_dataset = KnotDataset(dirname, file_path)

    knot_dataloader = DataLoader(knot_dataset, batch_size = Nbeads, shuffle = False)

    for batch in knot_dataloader:

        batch_unbind = torch.unbind(batch) # can maybe see if there is a method to have batch unbound initially?
        batch_stack = torch.stack(batch_unbind, dim = 1)

        batch = torch.reshape(batch_stack, (Nbeads, n_col))


    if dtype == "XYZ":

        for batch in knot_dataloader:

            batched_func_mean = torch.func.vmap(lambda x: x - torch.mean(x, 0), in_dims = 0)
            batch = batched_func_mean(batch)

    
    return batch

def split_train_test_validation(dataset, train_size, test_size, val_size):

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    return train_dataset, test_dataset, val_dataset




    
