# torch modules
import torch
from torch.utils.data import ConcatDataset, Subset
import numpy as np

# lightning modules
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# sklearn modules
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# scipy modules
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import shapiro

# code modules
from helper import *
from loader import *
from ml_models import *  # DT models
from nn_models import *
from nn_generative_models import *
from nn_concept_models import *
from analysis import *
from data_generation import *


device = torch.device("cpu")
def main():

    properties = {"dowker", "jones", "quantumA2", "HOMFLY"}
    datasets = []

################## <--classic classification problem + reconstruction--> ###################

    if pdct == "class": # used for doing a std classify problem vs. prediction problem (only Sig2XYZ right now)
        for i, knot in enumerate(knots): 
            indicies = np.arange(0, len_db) # first len_db
            if mode == "conditional": # generate tensors for conditional labels
                datasets.append(Subset(CondKnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i), indicies))
            else:
                datasets.append(Subset(KnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i), indicies))
            ##on cluster use below:
            #datasets.append(Subset(KnotDataset(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together

        print(dataset[0])

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

        elif dtype == "2DSIGWRITHE":
            in_layer = (Nbeads, Nbeads)
            print(in_layer)

        elif dtype == "SIGWRITHE":
            in_layer = (Nbeads, 1) # specify input layer (Different for sigwrithe and xyz)
            print(in_layer)

        out_layer = len(knots)

        cond_layer = (1, 5)

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
            generate(input_shape=in_layer, latent_dims = 2, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs, model = model) # note model here is standard FFNN used for predicting generated knot type

        if mode == "conditional":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            loss_fn = nn.MSELoss() 
            optimizer = "adam"
            conditional_generate(input_shape=in_layer, cond_shape = cond_layer, latent_dims = 100, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs, model = model)

################## <--'invariant' problem : predict data from StA -> some invariant--> ###################

    elif pdct in properties: 

        # StS predict dowker unseen **wed 7th Feb**

        # Renzo ricca helicity 
        indicies = np.arange(0, len_db) # first 100000
        for i, knot in enumerate(knots): 
            #datasets.append(Subset(data_2_inv(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i, pdct), indicies))
            ##on cluster use below:
            datasets.append(Subset(data_2_inv(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i, pdct), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together

        ninputs = len(dataset) # total dataset length
        print(ninputs)
        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        ## unsupervised predictions

        #dataset_unsupervised= Subset(data_2_inv(master_knots_dir, "5_2", net, dtype, Nbeads, pers_len, 4, pdct), indicies)

        ##on cluster use below:
        dataset_unsupervised= (Subset(data_2_inv(os.path.join(master_knots_dir,"5_2",f"N{Nbeads}",f"lp{pers_len}"), "5_2", net, dtype, Nbeads, pers_len, 4, pdct), indicies))
        dont_train, test_dataset_singular, val_dataset_singular = split_train_test_validation(dataset_unsupervised, (int(len(dataset_unsupervised)*0.1)), (int(len(dataset_unsupervised)*0.7)), (int(len(dataset_unsupervised)*0.2)), bs)

        if dtype  == "XYZ":
            in_layer = (Nbeads, 3)

        elif dtype == "2DSIGWRITHE":
            in_layer = (Nbeads, Nbeads)

        elif dtype == "SIGWRITHE":
            in_layer = (Nbeads, 1) # input layer

        if pdct == "dowker":
            out_layer = 7 # max length of longest row in dowker_{knot_choice}

        elif pdct == "jones":
            out_layer = 20 #[10*2]

        elif pdct == "quantumA2":
            out_layer = 62 #[31*2]

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            loss_fn = nn.MSELoss()
            train(model, model_type, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset_singular, test_loader= test_dataset_singular, epochs = epochs)

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

    elif pdct == "concept":
        for i, knot in enumerate(knots):
            indicies = np.arange(0, len_db) # first len_db
            datasets.append(Subset(ConceptKnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i), indicies))

        dataset = ConcatDataset(datasets) # concatenate datasets together
        print(dataset[0])

        ninputs = len(dataset)
        print(ninputs)

        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        in_layer = (Nbeads, 1)
        concept_layer = (1, 1) # 5x1 tensor 
        out_layer = len(knots)

        if mode == "train":
            loss_fn = nn.CrossEntropyLoss()
            optimizer = "adam"
            train_with_bottleneck(input_shape=in_layer, concept_shape=concept_layer, output_shape=out_layer, loss_fn=loss_fn, optimizer=optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)


    elif pdct == "latent":

        dataset = LatentKnotDataset()
        print(dataset[0])
        ninputs = len(dataset)
        print(ninputs)

        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        in_layer = (2, 1)
        out_layer = len(knots)

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            train(model = model, model_type = model_type, loss_fn = loss_fn, optimizer = optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)


def train(model, model_type, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):

    if model_type == "NN":
        A = []
        neural = NN(model=model, loss=loss_fn, opt=optimizer)
        # neural = NN.load_from_checkpoint("../trained models/StA_standard_prediction_2_ls/checkpoints/epoch=31-step=4224.ckpt", model=model, loss=loss_fn, opt=optimizer)
        # neural = NN.load_from_checkpoint("../trained models/StA_standard_prediction_2ls_SQRGRN8/checkpoints/epoch=28-step=2320.ckpt", model=model, loss=loss_fn, opt=optimizer)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0005, patience=10, verbose=True, mode="min")
        trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
        trainer.fit(neural, train_loader, val_loader)
        trainer.test(dataloaders=test_loader)
        ## will need to add back to function arguments

    elif model_type == "DT":
        algorithm = DecisionTree(prob, train_loader, test_loader)
        tree_structure, importance, test_point, decision_path = algorithm.classification()
        analysis = Analysis(test_loader, algorithm, prob)
        analysis.DT_interpreter(tree_structure, importance, test_point, decision_path)

    elif model_type == "testing":
        data_gen = StA(prob, train_loader, test_loader)
        data_gen.calc_area()
    
    all_predicted = []
    all_y = []

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
    knot_labels = ["0_1", "3_1", "4_1", "5_1", "5_2"]
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=knot_labels).plot()
    plt.title("StA Knot Classification")
    plt.savefig("confusion_matrix.png")
    plt.close()

    latent_space_1 = np.linspace(4.2793527, -3.2264833, 100) # max for sqrgrn8 class  
    latent_space_2 = np.linspace(4.887735, -3.5318224, 100) # min for sqrgrn8 class

    # latent_space_1 = np.linspace(3.9112763, -3.3556507, 100) # max for 5 class
    # latent_space_2 = np.linspace(3.80764, -3.884432, 100) # min for 5 class
    Z = np.zeros((100, 100))

    for idx, i in enumerate(latent_space_1):
        for idy, j in enumerate(latent_space_2):
            z = torch.tensor([i, j], dtype=torch.float32)
            z = z.unsqueeze(0)
            _, predicted = torch.max(neural.forward(z).data, 1)
            Z[idx][idy] = predicted

    print(Z)

    plt.contourf(latent_space_1, latent_space_2, Z, cmap='viridis')
    marker_x_3_1 = []
    marker_y_3_1 = []
    # mid_y = np.linspace(1.0, -1.25, 10)
    # mid_x = np.linspace(-1.0, 1.25, 10)
    # for idx, i in enumerate(mid_y):
    #     marker_x_3_1.append(mid_x[idx])
    #     marker_y_3_1.append(mid_y[idx])
    mid_y = np.linspace(-1.0, 1.2, 5)
    mid_x = np.linspace(-0.3, 1.2, 5)
    for idx, i in enumerate(mid_y):
        marker_x_3_1.append(mid_x[idx])
        marker_y_3_1.append(mid_y[idx])

    # marker_x_4_1 = []
    # marker_y_4_1 = []
    # # # mid_y = np.linspace(-3.0, 2, 10)
    # # # mid_x = np.linspace(-1, 3, 10)
    # mid_y_2 = [i - 0.5 for i in mid_y]
    # mid_x_2 = [i + 0.5 for i in mid_x]
    # for idx, i in enumerate(mid_y_2):
    #     marker_x_4_1.append(mid_x_2[idx])
    #     marker_y_4_1.append(mid_y_2[idx])

    # marker_x_5_1 = []
    # marker_y_5_1 = []
    # # # mid_y = np.linspace(-3.0, 2, 10)
    # # # mid_x = np.linspace(-1, 3, 10)
    # mid_y_3 = [i + 0.5 for i in mid_y]
    # mid_x_3 = [i - 0.5 for i in mid_x]
    # for idx, i in enumerate(mid_y_3):
    #     marker_x_5_1.append(mid_x_3[idx])
    #     marker_y_5_1.append(mid_y_3[idx])
    marker_phase_change_x = [-0.7, -0.9]
    marker_phase_change_y = [-1.5, -1.75]
    marker_phase_change_x_post = [-0.95, -1.14]
    marker_phase_change_y_post = [-1.25, -1.55]
    # plt.scatter(marker_x_3_1, marker_y_3_1, color='red', marker='x')
    # plt.scatter(marker_phase_change_x, marker_phase_change_y, color='blue', marker='x')
    # plt.scatter(marker_phase_change_x_post, marker_phase_change_y_post, color='red', marker='x')
    # plt.scatter(marker_x_4_1, marker_y_4_1, color='blue', marker='x')
    # plt.scatter(marker_x_5_1, marker_y_5_1, color='red', marker='x')
    plt.colorbar()
    plt.title("2D latent space input (StA)")
    plt.xlabel("Latent space 1")
    plt.ylabel("Latent space 2")
    plt.show()

    

    #find max and min for 2 latent spaces - create matrix with colours for prediction
        
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
        
def train_with_bottleneck(input_shape, concept_shape, output_shape, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):

    neural = onlyconceptNN(input_shape=input_shape, concept_shape=concept_shape, output_shape=output_shape, loss=loss_fn, opt=optimizer)
    # neural = onlyconceptNN.load_from_checkpoint("lightning_logs/version_310/checkpoints/epoch=25-step=6500.ckpt", input_shape=input_shape, concept_shape=concept_shape, output_shape=output_shape, loss=loss_fn, opt=optimizer)
    #version 302

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
    knot_labels = ["0_1", "3_1", "4_1", "5_1", "5_2"]
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=knot_labels).plot()
    plt.title("2-Concept bottleneck model (global writhe + no. peaks)")
    plt.show()


    print(conf_mat)

    # print(f"prediction {z[0]}, true {y[0]}")
    # print(f"peaks {cp1[0]}, true {c1[0]}")
    # print(f"area {cp2[0]}, true {c2[0]}")


def generate(input_shape, latent_dims, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, model):

    # arguements to take, beta, VAE, problem, datatype, dimension

    # trained_model = Path(f"../trained models/neural nets/StA_VAE_{prob}")
    # if trained_model.is_dir() == True:

    if prob == "5Class" and dtype == "SIGWRITHE":
        neural = VariationalAutoencoder.load_from_checkpoint("../trained models/StA_VAE_2D_5_Class/checkpoints/epoch=155-step=39000.ckpt",input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer, beta=1)

    elif prob == "SQRGRN8":
        neural = VariationalAutoencoder.load_from_checkpoint("../trained models/StA_VAE_2D_SQRGRN8/checkpoints/epoch=163-step=41000.ckpt",input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer, beta=1)

    else:

        neural = VariationalAutoencoder(input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer, beta=0.0001)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
        trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
        trainer.fit(neural, train_loader, val_loader)
        trainer.test(dataloaders=test_loader)

    #neural = Autoencoder(input_shape = input_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer)

    analysis = Analysis(test_loader, neural, prob)
    e_s, e_l, t_s = analysis.generative_latent_space() # e_s is the encoded samples latent spaces
    
    saved_space = Path(f"../knot data/latent space {prob}/encoded_samples_{latent_dims}.csv")
    if saved_space.is_file() == False:
        e_s.to_csv(f'../knot data/latent space {prob}/encoded_samples_{latent_dims}.csv', index=False)
        e_l.to_csv(f'../knot data/latent space {prob}/encoded_labels_{latent_dims}.csv', index=False)

    if prob == "5Class":
        latent_space_1 = np.linspace(e_s.iloc[:, 0].max(), e_s.iloc[:, 0].min(), 100)
        latent_space_2 = np.linspace(e_s.iloc[:, 1].max(), e_s.iloc[:, 1].min(), 100)
        spacing_1 = (max(latent_space_1) - min(latent_space_1))/ 100
        spacing_2 = (max(latent_space_2) - min(latent_space_2))/ 100
        # latent_space_1 = np.linspace(3.9112763, -3.3556507, 100)
        # latent_space_2 = np.linspace(3.80764, -3.884432, 100)

    # plotter = analysis.dimensional_reduction_plot("PCA", encoded_samples=e_s, encoded_labels=e_l, latent_space=l_s, new_data=new_xyz, new_data_label=new_xyz_label)
    # plotter = analysis.dimensional_reduction_plot("TSNE", encoded_samples=e_s, encoded_labels=e_l, latent_space=l_s, new_data=d_s, new_data_label=new_sta_label)

    knot_predictor = False

    if prob == "SQRGRN8":
        latent_space_1 = np.linspace(4.2793527, -3.2264833, 100) # max for sqrgrn8 class  
        latent_space_2 = np.linspace(4.887735, -3.5318224, 100) # min for sqrgrn8 class

    Ar = np.zeros((100, 100))
    Pe = np.zeros((100, 100))
    Pe_sp = np.zeros((100, 100))
    In = np.zeros((100, 100))
    Ku = np.zeros((100, 100))
    Sk = np.zeros((100, 100))
    Pr = np.zeros((100, 100))
    Zcr = np.zeros((100, 100))

    # features to look at

    for idx, i in enumerate(latent_space_1):
        for idy, j in enumerate(latent_space_2):

            # check if valid in generated latent space + check most similar true latent space value
            # if not valid, skip -- worry about after sampling sorted
            z = torch.tensor([i, j], dtype=torch.float32)
            t = 1
            # for k in range(len(e_s)):
            #     if k%1000 == 0: # increase speed
            #         if e_s.iloc[:, 0][k] < z[0]+(spacing_1*2):
            #             if e_s.iloc[:, 0][k] > z[0]-(spacing_1*2):
            #                 if e_s.iloc[:, 1][k] < z[1]+(spacing_2*2): 
            #                     if e_s.iloc[:, 1][k] > z[1]-(spacing_2*2):
            #                         t+=1
            # print(t, idx, idy)

            if t>0: # pass through decoder
                z = z.unsqueeze(0)
                gen = neural.decoder(z).detach().numpy()
                gen = gen[0].flatten()
                indices = np.arange(0, len(gen), 1)

                # check corresponding StA index -> find XYZ

                # area 
                area = np.trapz(y=gen, x=indices)
                Ar[idx][idy] = area

                # peaks
                peaks, properties = find_peaks(gen, prominence=0.5)
                vals = properties['prominences']
                spread = np.std(peaks)
                Pe[idx][idy] = len(peaks)
                Pe_sp[idx][idy] = spread

                # inflection
                inflection = np.diff(np.sign(np.diff(gen))).nonzero()[0] 
                In[idx][idy] = len(inflection)

                # zero crossing rate
                mean = np.mean(gen)
                gen_norm = gen - mean # normalize
                zcs = 0
                for l in range(1, len(gen_norm)):
                    if gen_norm[l-1] * gen_norm[l] < 0:
                        zcs += 1
                Zcr[idx][idy] = zcs / len(gen_norm)

                # kurtosis
                kurt = kurtosis(gen_norm)
                Ku[idx][idy] = kurt

                # skewness
                skewness = skew(gen_norm)
                Sk[idx][idy] = skewness

                # periodicity
                indices_new = np.arange(0, 10*len(gen_norm), 1)
                fft_result = np.fft.fft(gen_norm)
                freqs = np.fft.fftfreq(len(gen_norm), indices_new[1] - indices_new[0])
                magnitude = fft_result.real ** 2 + fft_result.imag ** 2
                period = freqs[np.argmax(magnitude)]
                period = 1/period
                Pr[idx][idy] = period 

    plt.contourf(latent_space_1, latent_space_2, Ar, cmap='viridis')
    plt.colorbar()
    plt.title("2D latent space input (StA) vs. Area")
    plt.show()

    plt.contourf(latent_space_1, latent_space_2, Pe, cmap='viridis')
    plt.colorbar()
    plt.title("2D latent space input (StA) vs. No. Peaks")
    plt.show()

    plt.contourf(latent_space_1, latent_space_2, Ku, cmap='viridis')
    plt.colorbar()
    plt.title("2D latent space input (StA) vs. Kurtosis")
    plt.show()

    plt.contourf(latent_space_1, latent_space_2, Sk, cmap='viridis')
    plt.colorbar()
    plt.title("2D latent space input (StA) vs. Skewness")
    plt.show()

    plt.contourf(latent_space_1, latent_space_2, Pr, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label(r'Periodicity of StA $log(\frac{1}{\omega})$')  # Set the label for the color bar
    plt.title("2D latent space input (StA) vs. Periodicity")
    plt.xlabel(r'Latent space: $\hat{z_{1}}$')
    plt.ylabel(r'Latent space: $\hat{z_{2}}$')
    plt.show()

def conditional_generate(input_shape, cond_shape, latent_dims, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, model):

    if prob == "5Class" and dtype == "SIGWRITHE":
        neural = ConditionalVAE.load_from_checkpoint("../trained models/StA_Cond_VAE_5_Class/checkpoints/epoch=267-step=67000.ckpt",input_shape = input_shape, cond_shape = cond_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer, beta=1)

    else:
        neural = ConditionalVAE(input_shape = input_shape, cond_shape = cond_shape, latent_dims = latent_dims, loss=loss_fn, opt=optimizer, beta=1)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=True, mode="min")
        trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
        trainer.fit(neural, train_loader, val_loader)
        trainer.test(dataloaders=test_loader)

    analysis = Analysis(test_loader, neural, prob)
    e_s, e_l, l_s, d_s, new_sta_label = analysis.generative_latent_space() # e_s is the encoded samples latent spaces

    # plotter = analysis.dimensional_reduction_plot("PCA", encoded_samples=e_s, encoded_labels=e_l, latent_space=l_s, new_data=d_s, new_data_label=new_sta_label)


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

    main()
