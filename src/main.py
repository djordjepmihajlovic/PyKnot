# torch modules
import torch
from torch.utils.data import ConcatDataset, Subset
import numpy as np

# lightning modules
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# code modules
from helper import *
from loader import *
from ml_models import *  # KNN and DT models
from nn_models import *
from nn_generative_models import *
from analysis import *
from data_generation import *

# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
device = torch.device("cpu")
# print("No. GPUs Available: ", available_gpus)

def main():

    properties = {"dowker", "jones", "quantumA2", "HOMFLY"}
    datasets = []

################## <--classic classification problem + reconstruction--> ###################

    if pdct == "class": # used for doing a std classify problem vs. prediction problem (only Sig2XYZ right now)
        for i, knot in enumerate(knots): 
            indicies = np.arange(0, len_db) # first len_db
            #datasets.append(Subset(KnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i), indicies))
            ##on cluster use below:
            datasets.append(Subset(KnotDataset(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together

        ninputs = len(knots) * len_db
        ninputs = len(dataset)
        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)
        print(len(train_dataset))

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)
            print(in_layer)
        else:
            in_layer = (Nbeads, 1) # specify input layer (Different for sigwrithe and xyz)

        out_layer = len(knots)

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            print(model_type)
            train(model = model, model_type = model_type, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)
        
        if mode == "test":
            print("testing functionality moved to train mode, occurs after training instance")

        if mode == "generate":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            loss_fn = nn.MSELoss() 
            optimizer = "adam"
            generate(input_shape=in_layer, latent_dims = 10, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs, model = model) # note model here is standard FFNN used for predicting generated knot type

################## <--'invariant' problem : predict data from StA -> some invariant--> ###################

    elif pdct in properties: 

        indicies = np.arange(0, len_db) # first 100000
        for i, knot in enumerate(knots): 
            datasets.append(Subset(StA_2_inv(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i, pdct), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together

        ninputs = len(dataset) # total dataset length
        print(ninputs)
        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        ## unsupervised predictions

        # dataset_unsupervised= Subset(StA_2_inv(master_knots_dir, "5_2", net, dtype, Nbeads, pers_len, 4, pdct), indicies)
        # dont_train, test_dataset_singular, val_dataset_singular = split_train_test_validation(dataset_unsupervised, (int(len(dataset_unsupervised)*0.1)), (int(len(dataset_unsupervised)*0.7)), (int(len(dataset_unsupervised)*0.2)), bs)

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)
        else:
            in_layer = (Nbeads, 1) # input layer

        if pdct == "dowker":
            out_layer = 32 # max length of longest row in dowker_{knot_choice}
        elif pdct == "jones":
            out_layer = 20 #[10*2]

        elif pdct == "quantumA2":
            out_layer = 62 #[31*2]

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            loss_fn = nn.MSELoss()
            train(model, model_type, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)

        if mode == "generate":
            print("generate not available for invariant prediction; try: python main.py -m generate -pred class")


################## <--generative weighted reconstruction--> ###################

# nb. takes in XYZ data + StS data as weights and passes through specified encoder -> decoder architecture

# maybe deprecated we will see........

    elif pdct == "weighted": # used for doing a std classify problem vs. prediction problem (only Sig2XYZ right now)
        for i, knot in enumerate(knots): 
            datasets.append(WeightedKnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i))

        dataset = ConcatDataset(datasets) # concatenate datasets together
        indicies = np.arange(0, 100000)
        dataset = Subset(dataset, indicies)

        ninputs = len(dataset)
        print(ninputs)

        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        in_layer = (Nbeads, Nbeads)
        out_layer = (Nbeads, 3)

        if mode == "generate":
            loss_fn = nn.MSELoss() 
            optimizer = "adam"
            generate_with_attention(input_shape=in_layer, output_shape=out_layer, latent_dims = 10, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)


def train(model, model_type, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):

    if model_type == "NN":
        A = []
        neural = NN(model=model, loss=loss_fn, opt=optimizer)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0005, patience=10, verbose=True, mode="min")
        trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
        trainer.fit(neural, train_loader, val_loader)
        trainer.test(dataloaders=test_loader)
        ## will need to add back to function argument
        with torch.no_grad():
            for x, y in test_loader:
                z = neural.forward(x)

        print(z[0])
        print(y[0])


    elif model_type == "DT":
        algorithm = DecisionTree(prob, train_loader, test_loader)
        tree_structure, importance, test_point, decision_path = algorithm.classification()
        analysis = Analysis(test_loader, algorithm, prob)
        analysis.DT_interpreter(tree_structure, importance, test_point, decision_path)

    elif model_type == "testing":
        data_gen = StA(prob, train_loader, test_loader)
        data_gen.calc_area()

    # below is rough analysis exploring shap values
    # for x,y in train_loader:
    #     ti = x
    #     dat = neural.forward(x)
    #     break

    # for x,y in test_loader:
    #     tr = x
    #     vals = y
    #     dat = neural.forward(x)
    #     break

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
        
# def check_knot(knot_construct, model, loss_fn, optimizer):



def generate(input_shape, latent_dims, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, model):

    # neural = VariationalAutoencoder(input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer, beta=1)
    neural = Autoencoder(input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer)

    device = torch.device('cpu')
    # neural.to(device) # ensure that RNN works on CPU

    ## pre-trained model StA VAE
    # neural = Autoencoder.load_from_checkpoint("../trained models/AE_01.ckpt",input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer)
    # neural = VariationalAutoencoder.load_from_checkpoint("../trained models/StA_VAE_5_Class/checkpoints/epoch=79-step=20000.ckpt",input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer, beta=1)
    ## want to set up callable JSON file with completed training (so dont have to constantly retrain...)
    ## will take in input_shape, latent_dims, loss_fn, optimizer etc, find if trained model exists; if not run neural
    ## check trainedmodels.json for structure...

    ## comment below out if pre-trained model

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=30, verbose=True, mode="min")
    trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
    trainer.fit(neural, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    # data_ideal_3_1 = torch.Tensor(np.loadtxt("../knot data/3_1_ideal_StA.csv", usecols=(2,)))
    # data_ideal_3_1 = data_ideal_3_1.view(-1, Nbeads, 1)
    # print(data_ideal_3_1)
    # neural.encoder.eval()
    # with torch.no_grad():
    #     z, dev = neural.encoder(data_ideal_3_1)
    #     print(z)

    analysis = Analysis(test_loader, neural, prob)
    e_s, e_l, l_s, new_xyz, new_xyz_label = analysis.generative_latent_space() # e_s is the encoded samples latent spaces

    plotter = analysis.dimensional_reduction_plot("none", encoded_samples=e_s, encoded_labels=e_l, latent_space=l_s, new_data=new_xyz, new_data_label=new_xyz_label)

    # mini_ls = []
    # maxi_ls = []
    # for i in range(0, 10):
    #     min, max = analysis.latent_space_params(e_s, i)
    #     mini_ls.append(min)
    #     maxi_ls.append(max)

    # latent_dim_values = []

    # for idx, i in enumerate(mini_ls):
    #     latent_dim_range = np.linspace(mini_ls[idx], maxi_ls[idx], 5)
    #     latent_dim_values.append(latent_dim_range)


    # want to have s.t. each graph has its predicted knot type
        # predicts type of generated knot from VAE using pre-trained model (99.2% accuracy)
    # knot_predictor = NN.load_from_checkpoint("../trained models/StA_standard_predicition/checkpoints/epoch=86-step=21750.ckpt", model=model, loss=nn.CrossEntropyLoss, opt=optimizer)

    # val = 0
    # latent_dim_values[0] = np.linspace(800, 850, 10) #true = 833.0503 (1), 841.9703 (5)
    # x = []
    # y = []
    # z = []
    # for i in latent_dim_values[0]:
    #     l_s = [i, 790.7610, -437.7817, -865.9095, i, 187.5121, -1747.0660, -2124.1187, -787.0219, 805.7379]
    #     # z = analysis.latent_space_generation(l_s, 1, knot_predictor, val)
    #     x1, y1, z1 = analysis.latent_space_generation_XYZ(l_s, 1, val)
    #     val += 1
    #     x.append(x1)
    #     y.append(y1)    
    #     z.append(z1)

    # fig = plt.figure()  
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x[0], y[0], z[0], color='blue', alpha=0.5)
    # ax.plot(x[5], y[5], z[5], color='green', alpha=0.5)
    # ax.plot(x[9], y[9], z[9], color='red', alpha=0.5)
    # plt.show()

    

    # for i in latent_dim_values[1]:
    #     l_s = [0.0, i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     z = analysis.latent_space_generation(l_s, 2, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[2]:
    #     l_s = [0.0, 0.0, i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     z = analysis.latent_space_generation(l_s, 3, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[3]:
    #     l_s = [0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     z = analysis.latent_space_generation(l_s, 4, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[4]:
    #     l_s = [0.0, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     z = analysis.latent_space_generation(l_s, 5, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[5]:
    #     l_s = [0.0, 0.0, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, 0.0]
    #     z = analysis.latent_space_generation(l_s, 6, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[6]:
    #     l_s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0]
    #     z = analysis.latent_space_generation(l_s, 7, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[7]:
    #     l_s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i, 0.0, 0.0]
    #     z = analysis.latent_space_generation(l_s, 8, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[8]:
    #     l_s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i, 0.0]
    #     z = analysis.latent_space_generation(l_s, 9, knot_predictor, val)
    #     val += 1

    # for i in latent_dim_values[9]:
    #     l_s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i]
    #     z = analysis.latent_space_generation(l_s, 10, knot_predictor, val)
    #     val += 1


def generate_with_attention(input_shape, output_shape, latent_dims, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):

    neural = AttentionAutoencoder(input_shape = input_shape, output_shape= output_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
    trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
    trainer.fit(neural, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    analysis = Analysis(test_loader, neural, prob)
    e_s, e_l, l_s, new_xyz, new_xyz_label = analysis.generative_latent_space()
    plotter = analysis.dimensional_reduction_plot("PCA", encoded_samples=e_s, encoded_labels=e_l, latent_space=l_s, new_data=new_xyz, new_data_label=new_xyz_label)



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
    if pdct == "dowker":
        dtype = "SIGWRITHE"

    main()
