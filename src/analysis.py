import torch
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Analysis:
    def __init__(self, data, model):

        self.data = data
        self.model = model

    def generative_latent_space(self): # change in this for StS to XYZ

        encoded_samples = []
        for x, k in self.data:
            self.model.encoder.eval()
            with torch.no_grad():
                z = self.model.encoder(x)
            for idy, dims in enumerate(z):
                encoded_sample = {f"Enc. Variable {j}": enc for j, enc in enumerate(dims)}
                encoded_sample['label'] = k[idy]
                encoded_samples.append(encoded_sample)

        encoded_samples = pd.DataFrame(encoded_samples)

        original_sample = x[:1]

        encoded_labels = encoded_samples["label"].copy()
        encoded_samples = encoded_samples.drop("label",axis=1)

        latent_space_z = encoded_samples.values[:1].tolist()
        label_z = encoded_labels[:1].tolist()

        self.model.decoder.eval()

        with torch.no_grad():
            z = self.model.decoder(torch.Tensor(latent_space_z))

        return encoded_samples, encoded_labels, latent_space_z , z, label_z
    

    def dimensional_reduction_plot(self, method, encoded_samples, encoded_labels, latent_space, new_data, new_data_label):

        label_names=['0_1', '3_1', '4_1', '5_1', '5_2', '6_1', '6_2', '6_3']

        plt.subplot(1, 1, 1)
        plt.tight_layout()

        if method == "PCA":
            pca = PCA(n_components=2)
            encoded_samples_reduced_PCA = pca.fit_transform(encoded_samples)
            sns.scatterplot( x=encoded_samples_reduced_PCA[:,0], y=encoded_samples_reduced_PCA[:,1], hue=[label_names[l] for l in encoded_labels])

        elif method == "TSNE":
            tsne = TSNE(n_components=2)
            encoded_samples_reduced_TSNE = tsne.fit_transform(encoded_samples)
            sns.scatterplot( x=encoded_samples_reduced_TSNE[:,0], y=encoded_samples_reduced_TSNE[:,1], hue=[label_names[l] for l in encoded_labels])

        # print(f"The found latent space: {latent_space}")
        print(f"Correspinding to construction: {new_data} of a {new_data_label} knot")

        plt.suptitle(f"XYZ {method} of VAE latent space")
        plt.show()


    def StA_reconstruct(self, data, model):

        # run code over given dataset and generate a new point
        with torch.no_grad():
            for x,y in data:
                dat, mean, log_var = model.forward(x)
                break


        x_list = np.arange(0, 100)

        # true data 
        true = x[1].detach().numpy()
        #reconstructed data
        reconstruction = dat[1].detach().numpy()

        z = np.polyfit(x_list, reconstruction, 20)
        z = [item for sublist in z.tolist() for item in sublist]
        p = np.poly1d(z)

        plt.subplot(2, 1, 1)
        plt.plot(x_list, reconstruction, '.', x_list, p(x_list), '--')
        plt.xlabel('Bead index')
        plt.ylabel('Generated StA Writhe')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(x_list, true)
        plt.xlabel('Bead index')
        plt.ylabel('True StA Writhe')
        plt.grid()

        label = int(y[1])
        names = {0: "unknot", 1: "trefoil (3_1)", 2: "figure-8 (4_1)", 3: "pentafoil (5_1)", 4: "three twist (5_2)"}

        plt.suptitle(f"VAE StA writhe: {names[label]}")
        plt.show()


    def SHAP_analysis(self, data, model):

        for x,y in data:
            ti = x
            dat = model.forward(x)
            break

        for x,y in data:
            true_value_step = x
            value_ID_step = y
            prediction_step = model.forward(x)
            break

        # dual pred -----

        true_ID = value_ID_step[0].detach().numpy()
        true_value = true_value_step[0].detach().numpy()
        prediction = prediction_step[0].detach().numpy()

        x_list = np.arange(0, 100)

        z = np.polyfit(x_list, prediction, 15)
        z = [item for sublist in z.tolist() for item in sublist]
        p = np.poly1d(z)

        plt.subplot(2, 1, 1)
        plt.plot(x_list, prediction, '.', x_list, p(x_list), '--')
        plt.xlabel('Bead index')
        plt.ylabel('XYZ Predicted StA Writhe')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(x_list, true_value)
        plt.xlabel('Bead index')
        plt.ylabel('True StA Writhe')
        plt.grid()

        plt.suptitle(f"StA Writhe vs Bead Index (FFNN)")
        plt.show()

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


    
