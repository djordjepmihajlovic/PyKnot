# torch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np

# lightning modules
import pytorch_lightning as pl

import seaborn as sns
import matplotlib.pyplot as plt

################## <--FFNN--> ###################

# goal right now is to remodel the conceptNN to have a concept bottleneck
# models needed = g(x) -> c
#                = f(g(x)) -> y

class conceptFFNNModel(nn.Module):
    # define model structure i.e. layers
    def __init__(self, input_shape, concept_shape, output_shape):
        """nn.Module initialization --> builds neural network

        Args:
            input_shape (list): dim of input array
            output_shape (int): size of output array
            norm (bool): norm

        Returns:
            nn.Module
        """
        super(conceptFFNNModel, self).__init__()
        self.flatten_layer = nn.Flatten()
        
        # init layers

        self.dense_layer1 = nn.Linear(input_shape[0]*input_shape[1], 320)

        # hidden layers to FFNN
        self.dense_layer2 = nn.Linear(320, 320)
        self.dense_layer3 = nn.Linear(320, 320)
        self.dense_layer4 = nn.Linear(320, 320)

        # concept bottleneck layers
        self.bottleneck_layer1 = nn.Linear(320, concept_shape[0])
        self.bottleneck_layer2 = nn.Linear(320, concept_shape[0])

        # output layer
        self.output_layer = nn.Linear(concept_shape[0]+concept_shape[0], output_shape)

    def forward(self, x):
        x = self.flatten_layer(x)

        if hasattr(self, 'bn_layer'):
            x = self.bn_layer(x)

        x = F.relu(self.dense_layer1(x))
        x = F.relu(self.dense_layer2(x))
        x = F.relu(self.dense_layer3(x))
        x = F.relu(self.dense_layer4(x))

        x_concept1 = self.bottleneck_layer1(x)
        x_concept2 = self.bottleneck_layer2(x)

        x = torch.cat((x_concept1, x_concept2), 1)

        x = self.output_layer(x)

        return x_concept1.view(-1, 1, 1), x_concept2.view(-1, 1, 1), F.softmax(x, dim=1) 
    
################## <--FFNN--> ###################

class onlyconceptFFNNModel(nn.Module):
    # define model structure i.e. layers
    def __init__(self, input_shape, concept_shape, output_shape):
        """nn.Module initialization --> builds neural network

        Args:
            input_shape (list): dim of input array
            output_shape (int): size of output array
            norm (bool): norm

        Returns:
            nn.Module
        """
        super(onlyconceptFFNNModel, self).__init__()
        self.flatten_layer = nn.Flatten()
        
        # init layers

        self.dense_layer1 = nn.Linear(concept_shape[0]+concept_shape[0], 320)

        # hidden layers to FFNN
        self.dense_layer2 = nn.Linear(320, 320)
        self.dense_layer3 = nn.Linear(320, 320)
        self.dense_layer4 = nn.Linear(320, 320)

        # output layer
        self.output_layer = nn.Linear(320, output_shape)

    def forward(self, x):
        x = self.flatten_layer(x)

        if hasattr(self, 'bn_layer'):
            x = self.bn_layer(x)

        x = F.relu(self.dense_layer1(x))
        x = F.relu(self.dense_layer2(x))
        x = F.relu(self.dense_layer3(x))
        x = F.relu(self.dense_layer4(x))

        x = self.output_layer(x)

        return F.softmax(x, dim=1) 
        

################## <--Pytorch LightningModule for train/val/test config--> ###################
    
class conceptNN(pl.LightningModule):
    def __init__(self, input_shape, concept_shape, output_shape, loss, opt):
        """pl.LightningModule --> builds train/val/test 

        Args:
            model (nn.Module): neural network module
            loss (nn.loss): loss function
            opt (torch.optim): optimizer

        Returns:
            loss
        """
        super().__init__()
        self.model = conceptFFNNModel(input_shape=input_shape, concept_shape=concept_shape, output_shape=output_shape)
        self.loss_concept = nn.MSELoss()
        self.loss_classify = loss
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.0001)

    def forward(self, x):
        # apply model layers
        x_concept1, x_concept2, x = self.model(x)
        return x_concept1, x_concept2, x

    def training_step(self, batch, batch_idx, loss_name = 'train_loss'):
        # training
        x, c1, c2, y = batch
        z_concept1, z_concept2, z = self.forward(x)
        loss_prediction = self.loss_classify(z, y)
        loss_concept1 = self.loss_concept(z_concept1, c1)
        loss_concept2 = self.loss_concept(z_concept2, c2)
        loss = 0.001*(loss_concept1 + loss_concept2) + 5*(loss_prediction)

        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        # validation
        x, c1, c2, y = batch
        z_concept1, z_concept2, z = self.forward(x)
        loss_prediction = self.loss_classify(z, y)
        loss_concept1 = self.loss_concept(z_concept1, c1)
        loss_concept2 = self.loss_concept(z_concept2, c2)
        loss = 0.001*(loss_concept1 + loss_concept2) + 5*(loss_prediction)

        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def test_step(self, batch, batch_idx, loss_name ='test_loss'):
        #test
        x, c1, c2, y = batch
        z_concept1, z_concept2, z = self.forward(x)
        loss_prediction = self.loss_classify(z, y)
        loss_concept1 = self.loss_concept(z_concept1, c1)
        loss_concept2 = self.loss_concept(z_concept2, c2)
        loss = loss_prediction 

        # calculate acc

        # # std. label
        _, predicted = torch.max(z.data, 1) 
        test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

        ## concepts
        # true = 0
        # false = 0
        # predicted = torch.round(z)
        # el = (y-predicted)

        # for idx, i in enumerate(el):
        #     if torch.sum(i) == 0.0:
        #         true += 1
        #     else:
        #         false += 1
                # print(f"true: {y[idx]}")
                # print(f"predicted: {predicted[idx]}")

        # test_acc = true/(true+false)
        #test_acc = (torch.sum(el).item()/ (len(y)*1.0))**(1/2)

        # log outputs
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})

        # self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def configure_optimizers(self):
        return self.optimiser
    

################## <--(only concept) Pytorch LightningModule for train/val/test config--> ###################
    
class onlyconceptNN(pl.LightningModule):
    def __init__(self, input_shape, concept_shape, output_shape, loss, opt):
        """pl.LightningModule --> builds train/val/test 

        Args:
            model (nn.Module): neural network module
            loss (nn.loss): loss function
            opt (torch.optim): optimizer

        Returns:
            loss
        """
        super().__init__()
        self.model = onlyconceptFFNNModel(input_shape=input_shape, concept_shape=concept_shape, output_shape=output_shape)
        self.loss_concept = nn.MSELoss()
        self.loss_classify = loss
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.000001)

    def forward(self, x):
        # apply model layers
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx, loss_name = 'train_loss'):
        # training
        x, c1, c2, y = batch
        z = self.forward(torch.cat((c1, c2), 1) )
        loss_prediction = self.loss_classify(z, y)
        loss = loss_prediction 

        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        # validation
        x, c1, c2, y = batch
        z = self.forward(torch.cat((c1, c2), 1) )
        loss_prediction = self.loss_classify(z, y)
        loss = loss_prediction 


        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def test_step(self, batch, batch_idx, loss_name ='test_loss'):
        #test
        x, c1, c2, y = batch
        z = self.forward(torch.cat((c1, c2), 1) )
        loss_prediction = self.loss_classify(z, y)
        loss = loss_prediction 


        # calculate acc

        # # std. label
        _, predicted = torch.max(z.data, 1) 
        test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

        ## concepts
        # true = 0
        # false = 0
        # predicted = torch.round(z)
        # el = (y-predicted)

        # for idx, i in enumerate(el):
        #     if torch.sum(i) == 0.0:
        #         true += 1
        #     else:
        #         false += 1
                # print(f"true: {y[idx]}")
                # print(f"predicted: {predicted[idx]}")

        # test_acc = true/(true+false)
        #test_acc = (torch.sum(el).item()/ (len(y)*1.0))**(1/2)

        # log outputs
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})

        # predicted_np = predicted.cpu().numpy()
        # y_np = y.cpu().numpy()

        # # Calculate confusion matrix
        # confusion_mat = confusion_matrix(y_np, predicted_np)
        # print(confusion_mat)

        # self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return self.optimiser
    