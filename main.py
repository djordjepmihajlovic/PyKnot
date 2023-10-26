import os, csv
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from helper import *
from loader import *
from model import *

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from pytorch_lightning import Trainer

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("No. GPUs Available: ", available_gpus)

def main():

    knots = ["0_1", "3_1", "4_1", "5_1", "5_2"]
    fname = "XYZ_0_1.dat.nos"
    dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    Nbeads = 100
    dtype = "SIGWRITHE"
    n_col = len([0, 1, 2])
    net = "FFNN"
    pers_len = 10
    len_db = 200000
    mode = "train"
    norm = False
    bs = 256
    epochs = 5
    ch = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Code/PyKnot/"
    dtype_f = "SIGWRITHE"
    dtype_l = "XYZ"
    predict = "std"


    datasets = []
    if predict == "std":
        for i, knot in enumerate(knots): 
            datasets.append(KnotDataset(dirname, knot, net, dtype, Nbeads, pers_len, i))

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
            train(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)

            # train1(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, epochs = epochs, checkpoint_filepath = ch)
        
        if mode == "test":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm)
            test(model, loss_fn, optimizer, test_loader = test_dataset, epochs = epochs)

            # checkpoint = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Code/PyKnot/checkpoint_epoch_11.pt"
            # test(model, test_loader = test_dataset, checkpoint_filepath=checkpoint)

    elif predict == "Sig2XYZ":
        for i, knot in enumerate(knots): 
            datasets.append(Wr_2_XYZ(dirname, knot, net, dtype_f, dtype_l, Nbeads, pers_len, i))

        dataset = ConcatDataset(datasets) # concatenate datasets together
        ninputs = len(knots) * len_db
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, int(ninputs * (0.9)), int(ninputs * (0.075)), int(ninputs * (0.025)), bs)

        in_layer = (Nbeads, 1) 
        out_layer = 300

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm)
            train(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)

            #train1(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, epochs = epochs, checkpoint_filepath = ch)
        
        if mode == "test":
            model, loss_fn, optimizer = generate_model(net, in_layer, knots, norm)
            test(model, loss_fn, optimizer, test_loader = test_dataset, epochs = epochs)

            # checkpoint = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Code/PyKnot/checkpoint_epoch_11.pt"
            # test1(model, test_loader = test_dataset, checkpoint_filepath=checkpoint)

def train(model, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):
    
    neural = NN(model=model, loss=loss_fn, opt=optimizer)
    trainer = Trainer(max_epochs=epochs, limit_train_batches=250)  # steps per epoch = 250
    trainer.fit(neural, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

def test(model, loss_fn, optimizer, test_loader, epochs):
    neural = NN(model=model, loss=loss_fn, opt=optimizer)
    trainer = Trainer(max_epochs=epochs)
    trainer.test(dataloaders=test_loader)


def train1(model, loss_fn, optimizer, train_loader, val_loader, epochs, checkpoint_filepath):
    """Training function

    Args:
        model (nn.Module): Model to use during training
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of training epochs
        checkpoint_filepath (str): Filepath for saving checkpoints
    """

    # Setting up the datasets used during training
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)

    # Early Stopping Callback
    es = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    best_val_loss = float('inf')
    steps_per_epoch = 250

    # Fitting the model
    for epoch in tqdm(range(epochs)):
        model.train()
        #for inputs, labels in train_loader:
        for i, (inputs, labels) in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1) 
                # predicted = outputs
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_loss /= val_dataset_size
            accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.10f}")

            # Update learning rate based on validation loss
            es.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(outputs[0])
                save_checkpoint(model, optimizer, epoch, val_loss, accuracy, checkpoint_filepath)

    # Saving training history
    with open(os.path.join(checkpoint_filepath, "training_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ['loss', 'val_loss', 'accuracy'])
        w.writeheader()
        w.writerow({'loss': loss.item(), 'val_loss': val_loss, 'accuracy': accuracy})


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, os.path.join(checkpoint_filepath, f'checkpoint_epoch_{epoch}.pt'))

def test(model, test_loader, checkpoint_filepath):
    """Testing function for the models created

    Args:
        model (nn.Module): PyTorch model
        test_loader (DataLoader): Test data loader
        checkpoint_filepath (str): Filepath for loading the model checkpoint
        bs (int): Batch size
    """
    knots = ["0_1", "3_1", "4_1", "5_1", "5_2"]

    # Loading the model
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Defining the batch size for the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Evaluating the accuracy on the test dataset:")

    # Evaluate model
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    test_loss /= len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Use the network to predict values
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Getting the confusion matrix and saving
    cf_matrix = confusion_matrix(all_labels, all_predictions)
    print(cf_matrix)
    ch = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Code/PyKnot/"
    np.savetxt(os.path.join(ch, "conf_m.txt"), cf_matrix, fmt="%i")

    df_cm = pd.DataFrame(cf_matrix, index=[i for i in knots],
                         columns=[i for i in knots])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,cmap="gray_r")
    plt.show()


# if __name__ == "__main__":
#     args = get_params()
#     prob = args.problem
#     dtype =  args.datatype
#     adj = args.adjacent
#     norm = args.normalised
#     net = args.network
#     epochs = args.epochs
#     knots = get_knots(prob)
#     mode = args.mode 
#     Nbeads = int(args.nbeads)
#     bs = int(args.b_size)
#     len_db = int(args.len_db)
#     master_knots_dir = args.master_knot_dir
#     pers_len = args.pers_len

#     checkpoint_filepath = f"NN_model"

#     main()

main()

