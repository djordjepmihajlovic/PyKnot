import os, csv

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt

from helper import get_knots, get_params, generate_model
from loader import load_dataset, split_train_test_validation
from model import NNmodel, EarlyStopper

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("No. GPUs Available: ", available_gpus)

def main():

    datasets = []
    for i, knot in enumerate(knots):
        datasets.append(load_dataset(os.path.join(master_knots_dir, knot, f"N{Nbeads}", f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i))

    dataset = RandomSampler(datasets)

    ninputs = len(knots) * len_db

    train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, int(ninputs * (0.9)), int(ninputs * (0.075)), int(ninputs * (0.025)))

    if mode == "train":

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)

        model = generate_model(net, in_layer, norm)

    if mode == "train":

        train(model, train_dataset = train_dataset, val_dataset = val_dataset, bs = bs)

def train(model, train_dataset, val_dataset, bs):

    train_dataset = DataLoader(train_dataset, batch_size = bs)
    val_dataset = DataLoader(val_dataset, batch_size = bs)



    optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001) # need to check optionals
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

    for epoch in range(epochs):
        for data in train_dataset:
            # data is a batch of features and labels
            X, y = data
            model.zero_grad()  # note: pytorch!
            output = net(X.view(-1, 28*28))
            loss = F.nll_loss(output, y) # note: one-hot vector [0, 0, 1, 0] use mean sq
            loss.backward() # backpropagation ***
            optimizer.step() # adjust weights

        print(loss)



if __name__ == "__main__":
    args = get_params()
    prob = args.problem
    dtype = args.datatype
    adj = args.adjacent
    norm = args.normalised
    net = args.network
    epochs = args.epochs
    knots = get_knots(prob)
    mode = args.mode 
    Nbeads = int(args.nbeads)
    bs = int(args.b_size)
    len_db = int(args.len_db)
    master_knots_dir = args.master_knot_dir
    pers_len = args.pers_len

    checkpoint_filepath = f"NN_model"

    main()
