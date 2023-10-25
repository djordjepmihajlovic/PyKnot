import os, csv
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from tqdm import tqdm

from helper import get_knots, get_params, generate_model
from loader import KnotDataset, split_train_test_validation
from model import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("No. GPUs Available: ", available_gpus)

def main():

    knots = ["0_1", "3_1", "4_1"]
    fname = "XYZ_0_1.dat.nos"
    dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/XYZ"
    Nbeads = 100
    dtype = "XYZ"
    n_col = len([0, 1, 2])
    net = "FFNN"
    pers_len = 0
    len_db = 200000
    mode = "test"
    norm = False
    bs = 256
    epochs = 50
    ch = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Code/PyKnot/"


    datasets = []
    for i, knot in enumerate(knots):
        datasets.append(KnotDataset(dirname, knot, net, dtype, Nbeads, pers_len, i))

    dataset = ConcatDataset(datasets) # concatenate datasets together
    sampler = RandomSampler(dataset) # create a random sampler for concatenated dataset (samples elements randomly)

    # Create a DataLoader
    # dataloader = DataLoader(dataset=dataset, batch_size=32, sampler=sampler)

    ninputs = len(knots) * len_db

    train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, sampler, int(ninputs * (0.9)), int(ninputs * (0.075)), int(ninputs * (0.025)))

    if mode == "train":

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)

        model, loss_fn, optimizer = generate_model(net, in_layer, knots, norm)

    if mode == "train":

        train(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, epochs = epochs, checkpoint_filepath = ch)

    if mode == "test":

        in_layer = (Nbeads, 3)

        model, loss_fn, optimizer = generate_model(net, in_layer, knots, norm)

        checkpoint = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Code/PyKnot/checkpoint_epoch_49.pt"

        test(model, test_loader = test_dataset, checkpoint_filepath=checkpoint)

def train(model, loss_fn, optimizer, train_loader, val_loader, epochs, checkpoint_filepath):
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
    # es = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

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
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss /= val_dataset_size
            accuracy = correct / total

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.10f}")

            # Update learning rate based on validation loss
            # es.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
    knots = ["0_1", "3_1", "4_1"]

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
