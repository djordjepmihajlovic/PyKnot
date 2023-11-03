# torch modules
import torch
from torch.utils.data import ConcatDataset
import numpy as np
from torch.autograd import Variable

# PyTorch Lightning for training 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# code modules
from helper import *
from loader import *
from model import *
from generative import *

# analysis
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("No. GPUs Available: ", available_gpus)

def main():

    datasets = []
    if predict == "class": # used for doing a std classify problem vs. prediction problem (only Sig2XYZ right now)
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
            print(net)
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm)
            train(model, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)
        
        if mode == "test":
            print("error -> test attr. in train fn at the moment.")

        if mode == "generate":
            loss_fn = nn.MSELoss() 
            optimizer = "adam"
            generate(input_shape=in_layer, latent_dims= 2, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)

    elif predict == "dual": # used for doing a 'dual' problem -> predict data a from b (eg. XYZ -> StA)
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


def train(model, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):
    
    neural = NN(model=model, loss=loss_fn, opt=optimizer)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0005, patience=10, verbose=True, mode="min")
    trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
    trainer.fit(neural, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    # below is rough analysis exploring shap values
    for x,y in train_loader:
        ti = x
        dat = neural.forward(x)
        break

    for x,y in test_loader:
        tr = x
        vals = y
        dat = neural.forward(x)
        break

    ## dual pred -----

    # true = y[0].detach().numpy()
    # prediction = dat[0].detach().numpy()

    # print(true)
    # print(prediction)

    # x_list = np.arange(0, 100)

    # z = np.polyfit(x_list, prediction, 15)
    # z = [item for sublist in z.tolist() for item in sublist]
    # p = np.poly1d(z)

    # plt.subplot(2, 1, 1)
    # plt.plot(x_list,prediction, '.', x_list, p(x_list), '--')
    # plt.xlabel('Bead index')
    # plt.ylabel('XYZ Predicted StA Writhe')
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(x_list, true)
    # # ax = plt.gca()
    # # yabs_max = abs(max(ax.get_ylim(), key=abs))
    # # ax.set_ylim([-yabs_max, yabs_max])
    # plt.xlabel('Bead index')
    # plt.ylabel('True StA Writhe')
    # plt.grid()

    # plt.suptitle(f"StA Writhe vs Bead Index (FFNN)")
    # plt.show()

    ## regular shap -----

    # explainer = shap.DeepExplainer(neural, Variable(ti))
    # shap_values = explainer.shap_values(Variable(tr))

    # # shap value gives the amount that a given feature has contributed to the determination of a prediction
    # # so shap value can be amount that writhe at bead index (x) has contributed to class prediction (3_1) for example.

    # unknot = []
    # trefoil = []
    # four_one = []
    # five_one = []
    # five_two = []

    # unk_sta = []
    # tref_sta = []
    # four_sta = []
    # fivei_sta = []
    # fiveii_sta = []

    # for idx, elem in enumerate(dat):
    #     check = torch.argmax(elem)
    #     if check == 0:
    #         unknot.append(shap_values[0][idx])
    #         unk_sta.append(tr[idx])
    #         unk = idx
    #     elif check == 1:
    #         trefoil.append(shap_values[1][idx]) # shap_values for specific class calc. in this case trefoil
    #         tref_sta.append(tr[idx]) # list of sta data corresponding to predicted trefoils
    #         iii = idx
    #     elif check ==2:
    #         four_one.append(shap_values[2][idx])
    #         four_sta.append(tr[idx])
    #         iv = idx
    #     elif check == 3:
    #         five_one.append(shap_values[3][idx])
    #         fivei_sta.append(tr[idx])
    #         v = idx
    #     elif check == 4:
    #         five_two.append(shap_values[4][idx])
    #         fiveii_sta.append(tr[idx])
    #         v2 = idx

    # x_list = np.arange(0, 100)

    # plt.subplot(2, 1, 1)
    # plt.plot(x_list, tr[v2])
    # plt.xlabel('Bead index')
    # plt.ylabel('StA Writhe')
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(x_list, shap_values[0][v2], label = "-0_1")
    # plt.plot(x_list, shap_values[1][v2], label = "-3_1")
    # plt.plot(x_list, shap_values[2][v2], label = "-4_1")
    # plt.plot(x_list, shap_values[3][v2], label = "-5_1")
    # plt.plot(x_list, shap_values[4][v2], label = "-5_2")
    # plt.legend()
    # ax = plt.gca()
    # yabs_max = abs(max(ax.get_ylim(), key=abs))
    # ax.set_ylim([-yabs_max, yabs_max])
    # plt.xlabel('Bead index')
    # plt.ylabel('Specific SHAP importance')
    # plt.grid()

    # plt.suptitle(f"Predicition: (5_2) {dat[v2]}")
    # plt.show()

def generate(input_shape, latent_dims, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):

    neural = Autoencoder(input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
    trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
    trainer.fit(neural, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    # for x,y in test_loader:
    #     dat = neural.forward(x)
    #     break

    # x_list = np.arange(0, 100)

    # true = x[0].detach().numpy()
    # prediction = dat[0].detach().numpy()

    # print(true)
    # print(prediction)

    # z = np.polyfit(x_list, prediction, 20)
    # z = [item for sublist in z.tolist() for item in sublist]
    # p = np.poly1d(z)

    # plt.subplot(2, 1, 1)
    # plt.plot(x_list,prediction, '.', x_list, p(x_list), '--')
    # plt.xlabel('Bead index')
    # plt.ylabel('Generated StA Writhe')
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(x_list, true)
    # # ax = plt.gca()
    # # yabs_max = abs(max(ax.get_ylim(), key=abs))
    # # ax.set_ylim([-yabs_max, yabs_max])
    # plt.xlabel('Bead index')
    # plt.ylabel('True StA Writhe')
    # plt.grid()

    # label = int(y[0])
    # names = {0: "unknot", 1: "3_1", 2: "4_1", 3: "5_1", 4: "5_2"}

    # plt.suptitle(f"Autoencoded StA writhe {names[label]}")

    plot_latent(neural, train_loader)
    plt.suptitle("StA latent space - autoencoder")

    plt.show()


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to('cpu'))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y)
        if i > num_batches:
            plt.colorbar()
            break


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
    if predict == "dual":
        dtype_f = "XYZ"
        dtype_l = "SIGWRITHE"

    main()

