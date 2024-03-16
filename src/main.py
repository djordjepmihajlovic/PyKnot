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
from scipy.stats import kurtosis
from scipy.stats import skew

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

    properties = {"dowker", "jones", "quantumA2", "HOMFLY", "v2", "v3"}
    datasets = []
################## <--classic classification problem + reconstruction--> ###################

    if pdct == "class" or pdct == "combinatoric": # used for doing a std classify problem, note combinatoric tries learn multiplications of data.
        for i, knot in enumerate(knots): 
            indicies = np.arange(0, len_db) # first len_db
            if mode == "conditional": # generate tensors for conditional labels
                datasets.append(Subset(CondKnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i), indicies))
            else:
                datasets.append(Subset(KnotDataset(master_knots_dir, knot, net, dtype, Nbeads, pers_len, i), indicies))
            ##on cluster use below:
                # datasets.append(Subset(KnotDataset(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i), indicies))

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

        print(torch.size(dataset))

        ninputs = len(dataset) # total dataset length
        print(ninputs)
        train_len = int(ninputs * (0.9))
        test_len = int(ninputs * (0.075))
        val_len = ninputs - (train_len + test_len)
        train_dataset, test_dataset, val_dataset = split_train_test_validation(dataset, train_len, test_len, val_len, bs)

        ## unsupervised predictions

        #dataset_unsupervised= Subset(data_2_inv(master_knots_dir, "5_2", net, dtype, Nbeads, pers_len, 4, pdct), indicies)

        ##on cluster use below:
        # dataset_unsupervised= (Subset(data_2_inv(os.path.join(master_knots_dir,"5_2",f"N{Nbeads}",f"lp{pers_len}"), "5_2", net, dtype, Nbeads, pers_len, 4, pdct), indicies))
        # dont_train, test_dataset_singular, val_dataset_singular = split_train_test_validation(dataset_unsupervised, (int(len(dataset_unsupervised)*0.1)), (int(len(dataset_unsupervised)*0.7)), (int(len(dataset_unsupervised)*0.2)), bs)

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

        elif pdct == "v2":
            out_layer = 1

        elif pdct == "v3":
            out_layer = 1

        if mode == "train":
            model, loss_fn, optimizer = generate_model(net, in_layer, out_layer, norm, pdct)
            loss_fn = nn.MSELoss()
            train(model, model_type, loss_fn, optimizer, train_loader = train_dataset, val_loader = val_dataset, test_loader= test_dataset, epochs = epochs)

        if mode == "generate":
            print("generate not available for invariant prediction; try: python main.py -m generate -pred class")


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
        neural = NN(model=model, loss=loss_fn, opt=optimizer, predict=pdct)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0005, patience=10, verbose=True, mode="min")
        trainer = Trainer(max_epochs=epochs, limit_train_batches=250, callbacks=[early_stop_callback])  # steps per epoch = 250
        trainer.fit(neural, train_loader, val_loader)
        trainer.test(dataloaders=test_loader)


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
        plt.title("StS Knot Classification")
        plt.savefig(f"confusion_matrix_{prob}.png")
        plt.close()

        # Choose an input image and its corresponding label

        analysis = Analysis(data=test_loader, model=neural, prob=prob)
        analysis.saliency_map(knots=knots)


    if pdct == "latent":

        e_s_data = np.loadtxt(f'../knot data/latent space {prob}/encoded_samples_2.csv', delimiter=',', dtype=np.float32)
        latent_space_1 = np.linspace(np.max(e_s_data[:, 0]), np.min(e_s_data[:, 0]), 100)
        latent_space_2 = np.linspace(np.max(e_s_data[:, 1]), np.min(e_s_data[:, 1]), 100)

        Z = np.zeros((100, 100))

        for idx, i in enumerate(latent_space_1):
            for idy, j in enumerate(latent_space_2):
                z = torch.tensor([i, j], dtype=torch.float32)
                z = z.unsqueeze(0)
                _, predicted = torch.max(neural.forward(z).data, 1)
                Z[idx][idy] = predicted

        plt.contourf(latent_space_1, latent_space_2, Z, cmap='viridis')

        # investigation in 3_1 space
        marker_x_3_1 = []
        marker_y_3_1 = []
        mid_y = np.linspace(-1.0, 1.2, 5)
        mid_x = np.linspace(-0.3, 1.2, 5)
        for idx, i in enumerate(mid_y):
            marker_x_3_1.append(mid_x[idx])
            marker_y_3_1.append(mid_y[idx])

        plt.scatter(marker_x_3_1, marker_y_3_1, color='red', marker='x')
        plt.colorbar()
        plt.title("2D latent space input (StA)")
        plt.xlabel("Latent space 1")
        plt.ylabel("Latent space 2")
        plt.show()

        
def train_with_bottleneck(input_shape, concept_shape, output_shape, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs):

    neural = conceptNN(input_shape=input_shape, concept_shape=concept_shape, output_shape=output_shape, loss=loss_fn, opt=optimizer)
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

    plot_dim_reduction = "None"

    if plot_dim_reduction == "PCA":
        plotter = analysis.dimensional_reduction_plot("PCA", encoded_samples=e_s, encoded_labels=e_l)
    elif plot_dim_reduction == "TSNE":
        plotter = analysis.dimensional_reduction_plot("TSNE", encoded_samples=e_s, encoded_labels=e_l)

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

    plt.scatter(e_s.iloc[:, 0], e_s.iloc[:, 1], c=e_l, cmap='viridis', s=10, alpha=0.5)
    mid_y = np.linspace(-0.8, 0.8, 5)
    mid_x = np.linspace(-0.75, 1.25, 5)
    point_x = np.zeros((5, 1))
    point_y = np.zeros((5, 1))
    for idx, i in enumerate(mid_y):
        plt.scatter(mid_x[idx], mid_y[idx], color='red', marker='x')
        dist_init = 100
        # find closest true point
        for k in range(len(e_s)):
            dist = ((abs(e_s.iloc[k, 0] - mid_x[idx]))**2 + (abs(e_s.iloc[k, 1] - mid_y[idx]))**2)**(1/2)
            if dist < dist_init:  
                point_x[idx] = e_s.iloc[k, 0]
                point_y[idx] = e_s.iloc[k, 1]
                dist_init = dist

        plt.scatter(point_x[idx], point_y[idx], color='green', marker='x')
    plt.show()

    extrapolate_features = True
    if extrapolate_features == True:
        peak_points = []
        marker_x_3_1 = []
        marker_y_3_1 = []
        corresponding_index = []
        mid_y = np.linspace(-1.0, 1.2, 5)
        mid_x = np.linspace(-0.3, 1.2, 5)
        for idx, i in enumerate(mid_y):
            dist_init = 100
            # find closest true point (specifically for 3_1 class)
            for k in range(len(e_s)):
                if e_l.iloc[k] == 1: # 3_1 class
                    dist = ((e_s.iloc[k, 0] - mid_x[idx])**2 + (e_s.iloc[k, 1] - mid_y[idx])**2)**(1/2)
                    if dist < dist_init:  
                        point = [e_s.iloc[k, 0], e_s.iloc[k, 1]]
                        index_point = k 
                        dist_init = dist
            
            prediction, peaks = analysis.latent_space_generation(mid_x[idx], mid_y[idx], t_s[index_point])
            peak_points.append(peaks)

            print(f"est. [{mid_x[idx]}, {mid_y[idx]}] -> {point} ")
            corresponding_index.append(index_point)

    # have to find unshuffled index in base dataset -> to find the corrseponding XYZ
    # this is quite a silly way to do it
    data_3_1_StA = np.loadtxt(f"/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/SIGWRITHE/3DSignedWrithe_3_1.dat.lp{pers_len}.dat.nos", usecols=(2,))
    data_3_1_StA = torch.tensor(data_3_1_StA, dtype=torch.float32)
    data_3_1_StA = data_3_1_StA.view(-1, Nbeads, 1)
    choice = 0
    t_s = np.array(t_s[corresponding_index[choice]]).flatten()
    for idx, i in enumerate(data_3_1_StA):
        test = i.detach().numpy().flatten()
        if (t_s==test).all():
            print(idx)
            break

    data_3_1_XYZ = np.loadtxt(f"/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/XYZ/XYZ_3_1.dat.nos", usecols=(0, 1, 2))
    data_3_1_XYZ = data_3_1_XYZ.reshape(-1, Nbeads, 3)

    data_point = data_3_1_XYZ[idx]

    fig = plt.figure()
    ax = plt.axes(projection='3d') 

    ## plot in 3D 
    ax.plot3D(data_point[:, 0], data_point[:, 1], data_point[:, 2], 'o-')
    ax.plot3D(data_point[peak_points[choice], 0], data_point[peak_points[choice], 1], data_point[peak_points[choice], 2], 'o', color='red')
    plt.show()

    test_projections = False
    if test_projections == True:
        proj_no = np.linspace(0, 1, 5)
        no_crossings = []
        for i in proj_no:
            projection_matrix = np.array([[1-i, i, 0], [0, 1, 0], [0, 0, 0]])
            projected = np.dot(data_point, projection_matrix)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(projected[:, 0], projected[:, 1], projected[:, 2], 'gray')
            plt.show()

            print(projected)
            # find projection with least number of crossings
            crossings = []
            for i in range(0, 100):
                crossing_per_segment = 0
                vec_x1 = projected[i, 0] 
                vec_x2 = projected[(i+1)%100, 0] 
                vec_y1 = projected[i, 1]
                vec_y2 = projected[(i+1)%100, 1]
                vec_z1 = projected[i, 2]
                vec_z2 = projected[(i+1)%100, 2]

                for j in range(0, 100): # looping forward
                    if j!= i and j!=(i+1)%100 and j!=(i-1)%100:
                        vec_x3 = projected[j%100, 0] 
                        vec_x4 = projected[(j+1)%100, 0] 
                        vec_y3 = projected[j%100, 1]
                        vec_y4 = projected[(j+1)%100, 1] 
                        vec_z3 = projected[j%100, 2]
                        vec_z4 = projected[(j+1)%100, 2]

                        # linear alg.
                        t_xy = ((vec_x1-vec_x3)*(vec_y3-vec_y4) - (vec_y1-vec_y3)*(vec_x3-vec_x4))/((vec_x1-vec_x2)*(vec_y3-vec_y4) - (vec_y1-vec_y2)*(vec_x3-vec_x4))
                        s_xy = ((vec_x1-vec_x3)*(vec_y1-vec_y2) - (vec_y1-vec_y3)*(vec_x1-vec_x2))/((vec_x1-vec_x2)*(vec_y3-vec_y4) - (vec_y1-vec_y2)*(vec_x3-vec_x4))
                        
                        if 0<=s_xy<=1:
                            if 0<=t_xy<=1:
                                crossings.append([i,j])
            # works now
            no_crossings.append(len(crossings))
        print(no_crossings)


    # features to look at
    gen_features = False
    if gen_features == True:

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
