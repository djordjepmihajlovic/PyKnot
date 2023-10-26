import torch
from torch.utils.data import ConcatDataset

from pytorch_lightning import Trainer

from helper import *
from loader import *
from model import *

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("No. GPUs Available: ", available_gpus)

def main():

    datasets = []
    if predict == "std": # used for doing a std classify problem vs. prediction problem (only Sig2XYZ right now)
        for i, knot in enumerate(knots): 
            datasets.append(KnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i))

        dataset = ConcatDataset(datasets) # concatenate datasets together
        ninputs = len(knots) * len_db
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, int(ninputs * (0.9)), int(ninputs * (0.075)), int(ninputs * (0.025)), bs)

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)
        else:
            in_layer = (Nbeads, 1) # specify input layer (Different for sigwrithe and xyz)

        out_layer = len(knots)

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm)
            train(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs, knots=knots)
        
        if mode == "test":
            print("error -> test attr. in train fn at the moment.")

    elif predict == "Sig2XYZ":
        for i, knot in enumerate(knots): 
            datasets.append(Wr_2_XYZ(master_knots_dir, knot, net, dtype_f, dtype_l, Nbeads, pers_len, i))

        dataset = ConcatDataset(datasets) # concatenate datasets together
        ninputs = len(knots) * len_db
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, int(ninputs * (0.9)), int(ninputs * (0.075)), int(ninputs * (0.025)), bs)

        if dtype_f  == "XYZ":
            in_layer = (Nbeads, 3)
            out_layer = 100
        else:
            in_layer = (Nbeads, 1) # specify input layer (Different for sigwrithe and xyz)
            out_layer = 300

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm)
            train(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)
        
        if mode == "test":
            print("error -> test attr. in train fn at the moment.")


def train(model, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, knots):
    
    neural = NN(model=model, loss=loss_fn, opt=optimizer)
    trainer = Trainer(max_epochs=epochs, limit_train_batches=250)  # steps per epoch = 250
    trainer.fit(neural, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

if __name__ == "__main__":
    args = get_params()
    prob = args.problem
    dtype =  args.datatype
    adj = args.adjacent
    norm = args.normalised
    net = args.network
    epochs = args.epochs
    knots = get_knots(prob)
    mode = args.mode 
    Nbeads = int(args.nbeads)
    bs = int(args.b_size)
    len_db = int(args.len_db)
    master_knots_dir = args.master_knots_dir
    pers_len = args.pers_len
    predict = args.predictor
    if predict == "Sig2XYZ":
        dtype_f = "SIGWRITHE"
        dtype_l = "XYZ"

    main()

