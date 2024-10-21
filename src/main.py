# torch modules
import torch
from torch.utils.data import ConcatDataset, Subset
import numpy as np

# lightning modules
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# sklearn modules
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# code modules
from helper import *
from loader import *
from models.nn_models import *
from models.concept_models import *
from analysis import *


device = torch.device("cpu")
def main():

    properties = {"dowker", "v2", "v3", "v2v3"}
    datasets = []
################## <--classic classification problem + reconstruction--> ###################

    if pdct == "class": # used for doing a standard classification problem

        for i, knot in enumerate(knots): 
            indicies = np.arange(0, len_db) # first len_db
            #datasets.append(Subset(KnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i), indicies))
            ##on cluster use below:
            datasets.append(Subset(KnotDataset(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together

        ninputs = len(dataset)

        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)
            print(in_layer)

        elif dtype == "2DSIGWRITHE":
            in_layer = (Nbeads, Nbeads)
            print(in_layer)

        elif dtype == "SIGWRITHE":
            in_layer = (Nbeads, 1) # specify input layer (Different for sigwrithe and xyz)
            print(in_layer)

        out_layer = len(knots)

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            print(model)
            train(model = model, model_type = model_type, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)
        
        if mode == "test":
            print("testing functionality moved to train mode, occurs after training instance")


################## <--'invariant' problem : predict data from: (input -> some measure/invariant) --> ###################

    elif pdct in properties: 

        indicies = np.arange(0, len_db) # first 100000
        for i, knot in enumerate(knots): 
            #datasets.append(Subset(data_2_inv(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i, pdct), indicies))
            ##on cluster use below:
            datasets.append(Subset(data_2_inv(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i, pdct), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together
        print(dataset[0])

        ninputs = len(dataset) # total dataset length
        print(ninputs)
        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)

        elif dtype == "2DSIGWRITHE":
            in_layer = (Nbeads, Nbeads)

        elif dtype == "SIGWRITHE":
            in_layer = (Nbeads, 1) # input layer

        if pdct == "dowker":
            out_layer = 32 # max length of longest row in dowker_{knot_choice}

        elif pdct == "v2" or pdct == "v3":
            out_layer = 1

        elif pdct == "v2v3":
            out_layer = 2

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            print(model)
            loss_fn = nn.MSELoss()
            train(model, model_type, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)


################## <--concept problem : concept bottle neck model --> ###################

    elif pdct == "concept": # this class uses a pretrained model as its concept bottleneck
        for i, knot in enumerate(knots):
            indicies = np.arange(0, len_db) # note want len_db to be 10,000
            #datasets.append(Subset(data_2_inv(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i, pdct), indicies))
            ##on cluster use below:
            datasets.append(Subset(data_2_inv(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i, pdct), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together
        print(dataset[0])

        ninputs = len(dataset)
        print(ninputs)

        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        in_layer = (Nbeads, Nbeads)
        concept_layer = (1, 2) 
        out_layer = len(knots)

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, concept_layer, norm, "v2v3")
            print(model)
            loss_fn_bottleneck = nn.MSELoss()
            loss_fn_classify = nn.CrossEntropyLoss()
            optimizer = "adam"
            train_concept(model= model, input_shape=in_layer, concept_shape=concept_layer, output_shape=out_layer, loss_fn_bottleneck=loss_fn_bottleneck, loss_fn_classify=loss_fn_classify, optimizer=optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)


def train(model, model_type, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):

    if model_type == "NN":
        neural = NN(model=model, loss=loss_fn, opt=optimizer, predict=pdct)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0005, patience=10, verbose=True, mode="min")
        trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
        trainer.fit(neural, train_loader, val_loader)
        trainer.test(dataloaders=test_loader)
    
    all_predicted = []
    all_y = []

    if pdct == "class":

        with torch.no_grad():
            for x, y in test_loader:
                z = neural.forward(x)
            
                _, predicted = torch.max(z.data, 1) 
                test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

                predicted_np = predicted.cpu().numpy()
                y_np = y.cpu().numpy()

                # Accumulate predictions
                all_predicted.extend(predicted_np)
                all_y.extend(y_np)

        # Calculate confusion matrix over all batches
        conf_mat = confusion_matrix(all_y, all_predicted)

        # Display confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=knots).plot(include_values=False)
        if dtype == "SIGWRITHE":
            plt.title(f"{prob} StA Knot Classification")
        elif dtype == "2DSIGWRITHE":
            plt.title(f"{prob} StS Knot Classification")
        plt.savefig(f"confusion_matrix_{prob}.png")
        plt.close()

        # Choose an input image and its corresponding label

        #analysis = Analysis(data=test_loader, model=neural, prob=prob)
        #analysis.saliency_map(knots=knots)

    predictions = []
    true_vals = []
    true_vals_v2 = []
    true_vals_v3 = []
    labels = []
    if pdct != "class":
        if pdct == "v2" or pdct == "v3":
            with torch.no_grad():
                for x, y, c in test_loader:
                    z = neural.forward(x)

                    predicted_np = z.cpu().numpy()
                    true_vals_np = y.cpu().numpy()
                    labels_np = c.cpu().numpy()

                    # Accumulate predictions
                    predictions.extend(predicted_np)
                    true_vals.extend(true_vals_np)
                    labels.extend(labels_np)

            predictions = [item.item() for array in predictions for item in array]
            true_vals = [item.item() for array in true_vals for item in array]

            output_data = [[] for i in range(0, len(knots))]
            for idx, i in enumerate(labels):
                output_data[int(i)].append(predictions[idx])

            output_data_corr = [[] for i in range(0, len(knots))]
            for idx, i in enumerate(labels):
                output_data_corr[int(i)].append(true_vals[idx])

            for idx, i in enumerate(output_data):
                with open(f'vassiliev_{knots[idx]}_{pdct}_solo_predictions.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for item in i:
                        writer.writerow([item])

            for idx, i in enumerate(output_data_corr):
                with open(f'vassiliev_{knots[idx]}_{pdct}_solo_true.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for item in i:
                        writer.writerow([item])
        
        elif pdct == "v2v3":
            with torch.no_grad():
                for x, y1, y2, c in test_loader:
                    z = neural.forward(x)
                    y = torch.cat((y1, y2), 1)

                    predicted_np = z.cpu().numpy()
                    true_vals_np1 = y1.cpu().numpy()
                    true_vals_np2 = y2.cpu().numpy()
                    labels_np = c.cpu().numpy()

                    # Accumulate predictions
                    predictions.extend(predicted_np)
                    true_vals_v2.extend(true_vals_np1)
                    true_vals_v3.extend(true_vals_np2)
                    labels.extend(labels_np)

            predictions = [item.item() for array in predictions for item in array]
            true_vals_v2 = [item.item() for array in true_vals_v2 for item in array]
            true_vals_v3 = [item.item() for array in true_vals_v3 for item in array]

            output_data = [[] for i in range(0, len(knots))]
            for idx, i in enumerate(labels):
                output_data[int(i)].append(predictions[idx])

            output_data_corr = [[] for i in range(0, len(knots))]
            for idx, i in enumerate(labels):
                output_data_corr[int(i)].append([true_vals_v2[idx], true_vals_v3[idx]])

            for idx, i in enumerate(output_data):
                with open(f'vassiliev_{knots[idx]}_{pdct}_solo_predictions.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for item in i:
                        writer.writerow([item])

            for idx, i in enumerate(output_data_corr):
                with open(f'vassiliev_{knots[idx]}_{pdct}_solo_true.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    for item in i:
                        writer.writerow([item])


        
def train_concept(model, input_shape, concept_shape, output_shape, loss_fn_bottleneck, loss_fn_classify, optimizer, train_loader, val_loader, test_loader, epochs):

    #model_name = "/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/trained models/v2v3_5Class_RNN/checkpoints/epoch=480-step=84656.ckpt"
    model_name = "../PyKnot/trained models/v2v3_5Class_RNN/checkpoints/epoch=480-step=84656.ckpt"
    #model_name = "../trained models/v2v3_5Class_RNN/checkpoints/epoch=480-step=84656.ckpt"
    G_x = NN.load_from_checkpoint(model_name, model=model, loss=loss_fn_bottleneck, opt=optim.Adam(model.parameters(), lr=0.000001), predict="v2v3")


    neural = postmodelNN(model=model, input_shape=input_shape, concept_shape=concept_shape, output_shape=output_shape, loss_fn_bottleneck=loss_fn_bottleneck, loss_fn_classify=loss_fn_classify, G_x=G_x)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
    trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
    trainer.fit(neural, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    all_predicted = []
    all_y = []

    with torch.no_grad():
        for x, c1, c2, y in test_loader:
            # cp1, cp2, z = neural.forward(x)
            z = neural.forward(torch.cat((c1, c2), 1)) # only concepts
        
            _, predicted = torch.max(z.data, 1) 
            test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

            predicted_np = predicted.cpu().numpy()
            y_np = y.cpu().numpy()

            # Accumulate predictions
            all_predicted.extend(predicted_np)
            all_y.extend(y_np)

    # Calculate confusion matrix over all batches
    conf_mat = confusion_matrix(all_y, all_predicted)

    # Display confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=knots).plot(include_values=False)
    plt.savefig(f"confusion_matrix_{prob}.png")

    print(conf_mat)


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
    pdct = args.predictor
    model_type = args.model_type

    main()
