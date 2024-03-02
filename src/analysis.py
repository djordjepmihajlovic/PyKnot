import torch
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import find_peaks 

class Analysis:
    def __init__(self, data, model, prob):

        self.data = data
        self.model = model
        self.prob = prob

    def generative_latent_space(self): # change in this for StS to XYZ
        """
        Generate a latent space from the given data.

        Returns:
            np.array: The latent space.
        """

        encoded_samples = []
        true_sample = []

        for x, k in self.data: # three inputs for conditional
            self.model.encoder.eval()
            with torch.no_grad():
                z, std = self.model.encoder(x) # remember if using variational autoencoder need to add "dev"
                # z = self.model.reparameterization(mean, std) # will need to dbl check this
            for idy, dims in enumerate(z):
                encoded_sample = {f"Enc. Variable {j}": enc.item() for j, enc in enumerate(dims)}
                true_sample.append(x[idy].tolist())
                encoded_sample['label'] = k[idy].item()
                encoded_samples.append(encoded_sample)

        encoded_samples = pd.DataFrame(encoded_samples)

        encoded_labels = encoded_samples["label"].copy()
        encoded_samples = encoded_samples.drop("label",axis=1)

        return encoded_samples, encoded_labels, true_sample
    
    def latent_space_params(self, latent_space_df, col): # used for returning the min and max vals of the latent space per dimension

        min = latent_space_df[f'Enc. Variable {col}'].min()    
        max = latent_space_df[f'Enc. Variable {col}'].max()

        return min, max  
    
    def latent_space_generation_XYZ(self, latent_space, dim, val):
        self.model.decoder.eval()

        with torch.no_grad():
            z = self.model.decoder(torch.Tensor(latent_space))

        prediction = z.detach().numpy()[0]
        

        x= prediction[:,0]
        y= prediction[:,1]
        z= prediction[:,2]

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(x, y, z, color='blue', alpha=0.5)

        return x, y, z

    def latent_space_generation(self, latent_space_x, latent_space_y, true_sample):
        """
        Return a decoded sample from a given latent space point.

        Args:
            latent_space_x (float): The x coordinate of the latent space point.
            latent_space_y (float): The y coordinate of the latent space point.
            true_sample (np.array): The true sample.

        Returns:
            np.array: The decoded sample.
        """

        self.model.decoder.eval()
        with torch.no_grad():
            z = self.model.decoder(torch.tensor([latent_space_x, latent_space_y], dtype=torch.float32))

        x_list = np.arange(0, 100)

        prediction = z.detach().numpy()[0]

        print(f'{z} : predicted StA Writhe')

        z = np.polyfit(x_list, prediction, 20)
        z = [item for sublist in z.tolist() for item in sublist]
        p = np.poly1d(z)

        peaks, properties = find_peaks(np.array(true_sample).flatten(), prominence=0.1)
        vals = properties['prominences']


        sns.set_style("whitegrid") 
        for i in peaks:
            plt.scatter(i, prediction[i], marker='x', color='r')
        plt.plot(x_list,prediction, '.', x_list, p(x_list), '--')
        plt.plot(true_sample, label = "True StA Writhe")

        plt.ylim([-2.5, 2.5])
        plt.xlabel('Bead index')
        plt.ylabel(f'Generated StA Writhe.')
        plt.gca().tick_params(which="both", direction="in", right=True, top=True)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.tight_layout()
        plt.show()

        return prediction, peaks
# refactor for showing max vs min on one plot
    
    def latent_space_generation_maxmin(self, latent_space_1, latent_space_t, latent_space_2, dim, model, val):

        self.model.decoder.eval()

        with torch.no_grad():
            z1 = self.model.decoder(torch.Tensor(latent_space_1))
            z2 = self.model.decoder(torch.Tensor(latent_space_2))
            zt = self.model.decoder(torch.Tensor(latent_space_t))

        x_list = np.arange(0, 100)

        prediction_1 = z1.detach().numpy()[0]
        prediction_2 = z2.detach().numpy()[0]
        prediction_t = zt.detach().numpy()[0]

        if model == True:

            with torch.no_grad():
                z_new_1 = model.forward(z1)
                _, knot_type1 = torch.max(z_new_1.data, 1) 
                total = torch.sum(z_new_1.data)
                certainty_max = torch.max(z_new_1.data) / total

            with torch.no_grad():
                z_new_2 = model.forward(z2)
                _, knot_type2 = torch.max(z_new_2.data, 1) 
                total = torch.sum(z_new_2.data)
                certainty_min = torch.max(z_new_2.data) / total

            with torch.no_grad():
                z_new_t = model.forward(zt)
                _, knot_typet = torch.max(z_new_t.data, 1) 
                total = torch.sum(z_new_t.data)
                certainty_t = torch.max(z_new_t.data) / total

        knot_names= {0: "unknot", 1: "trefoil (3_1)", 2: "figure-8 (4_1)", 3: "pentafoil (5_1)", 4: "three twist (5_2)"}

        z_max = np.polyfit(x_list, prediction_1, 20)
        z_max = [item for sublist in z_max.tolist() for item in sublist]
        p_max = np.poly1d(z_max)

        z_min = np.polyfit(x_list, prediction_2, 20)
        z_min = [item for sublist in z_min.tolist() for item in sublist]
        p_min = np.poly1d(z_min)

        z_t = np.polyfit(x_list, prediction_t, 20)
        z_t = [item for sublist in z_t.tolist() for item in sublist]
        p_t = np.poly1d(z_t)




        sns.set_style("whitegrid") 
        plt.plot(x_list,prediction_t, '.', x_list, p_t(x_list), '--', alpha = 1)
        plt.plot(x_list,prediction_1, '.', x_list, p_max(x_list), '--', alpha = 0.5)
        plt.plot(x_list,prediction_2, '.', x_list, p_min(x_list), '--', alpha = 0.5)
        plt.ylim([-2.5, 2.5])
        plt.xlabel('Bead index')
        plt.ylabel(f'Generated StA Writhe.')
        plt.gca().tick_params(which="both", direction="in", right=True, top=True)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.tight_layout()
        # plt.savefig(f"latent_plots_2/3_1_l_s{val}.png")
        plt.show()


    def dimensional_reduction_plot(self, method, encoded_samples, encoded_labels):

        label_names=['0_1', '3_1', '4_1', '5_1', '5_2']

        plt.subplot(1, 1, 1)
        plt.tight_layout()

        if method == "PCA":
            pca = PCA(n_components=2)
            encoded_samples_reduced_PCA = pca.fit_transform(encoded_samples)
            sns.scatterplot( x=encoded_samples_reduced_PCA[:,0], y=encoded_samples_reduced_PCA[:,1], hue=[label_names[l] for l in encoded_labels])
            plt.title(f"StA {method} of 3-VAE, 10D latent space")
            plt.savefig(f"{method}.png")
            plt.close()

        elif method == "TSNE":
            tsne = TSNE(n_components=2)
            encoded_samples_reduced_TSNE = tsne.fit_transform(encoded_samples)
            sns.scatterplot( x=encoded_samples_reduced_TSNE[:,0], y=encoded_samples_reduced_TSNE[:,1], hue=[label_names[l] for l in encoded_labels])
            plt.title(f"StA {method} of 2-VAE, 3D latent space")
            plt.savefig(f"{method}.png")
            plt.close()


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

    def DT_interpreter(self, tree_structure, importance, test_point, decision_path):

        x = np.arange(0, 100)  
        node_index = 0
        features = []
        thresholds = []

        for i in range(tree_structure.node_count):
            # Check if the node is part of the decision path
            if decision_path[i] == 1:
                # Get feature index and threshold for the decision at this node
                feature_index = tree_structure.feature[node_index]
                threshold = tree_structure.threshold[node_index]
                
                # print(f"Node {node_index}: Feature {feature_index} <= {threshold}")
                features.append(feature_index)
                thresholds.append(threshold)
                
                # Determine the direction of the decision (left or right)
                decision = "left" if test_point[feature_index] <= threshold else "right"
                
                # Move to the next node based on the decision
                if decision == "left":
                    node_index = tree_structure.children_left[node_index]
                else:
                    node_index = tree_structure.children_right[node_index]

        # features = np.unique(np.abs(features))
        features, ind = np.unique(np.abs(features), return_index=True)

        test_point_DT = [i for idx,i in enumerate(test_point) if idx in features]

        ax = sns.barplot(x=x, y=importance, color='blue')
        ax.set_xticklabels([])

        plt.xlabel("Bead index")
        plt.ylabel("Relative importance")
        plt.title(f"Decision Tree Feature Importance {self.prob}")
        plt.gca().tick_params(which="both", direction="in", right=True, top=True)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.show()

        plt.scatter(features, test_point_DT, marker='x', label = "DT nodes", color='black')
        for i, txt in enumerate(ind):
            plt.annotate(f"{txt}", (features[i], test_point_DT[i]))
        plt.plot(x, test_point, '--', label = "Base")

        print(thresholds)

        table_data = [[f"Feature {features[i]}", f"{thresholds[i]:.1f}"] for i in range(len(features))]

        plt.legend()
        plt.xlabel("Bead index")
        plt.ylabel("StA Writhe")
        plt.gca().tick_params(which="both", direction="in", right=True, top=True)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.title("DT model +ve prediction")

        plt.show()


    def saliency_map(self):
        #we don't need gradients w.r.t. weights for a trained model
        for param in self.model.parameters():
            param.requires_grad = False
        
        #set model in eval mode
        self.model.eval()
        for x, y in self.data:
            x.requires_grad = True
            input_img = x
            preds = self.model(x)
            self.model.zero_grad()
            loss = torch.nn.CrossEntropyLoss()
            loss_cal = loss(preds, y)
            loss_cal.backward()
            saliency_map = x.grad.abs().max(1)[0]
        
        #plot image and its saleincy map
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(input_img[0].detach().numpy(), (1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(saliency_map[0].squeeze().cpu().numpy(), cmap=plt.cm.hot)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        plt.savefig("saliency_map.png")








        
        







    
