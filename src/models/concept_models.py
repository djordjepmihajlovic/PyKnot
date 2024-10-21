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

from models.nn_models import *

# goal 
# = g(x) -> c
# = f(g(x)) -> y
    
################## <--FFNN--> ###################

class f_c(nn.Module):
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
        super(f_c, self).__init__()
        self.flatten_layer = nn.Flatten()

        # init layers

        self.dense_layer1 = nn.Linear(concept_shape, 320)

        # hidden layers to FFNN
        self.dense_layer2 = nn.Linear(320, 320)
        self.dense_layer3 = nn.Linear(320, 320)
        self.dense_layer4 = nn.Linear(320, 320)

        # output layer
        self.output_layer = nn.Linear(320, output_shape)

    def forward(self, x):
        x = self.flatten_layer(x)

        x = F.relu(self.dense_layer1(x))
        x = F.relu(self.dense_layer2(x))
        x = F.relu(self.dense_layer3(x))
        x = F.relu(self.dense_layer4(x))

        x = self.output_layer(x)

        return F.softmax(x, dim=1) 
    

################## <--(only concept) Pytorch LightningModule for train/val/test config--> ###################
    
class postmodelNN(pl.LightningModule):
    def __init__(self, model, input_shape, concept_shape, output_shape, loss_fn_bottleneck, loss_fn_classify, G_x):
        """pl.LightningModule --> builds train/val/test 

        Args:
            model (nn.Module): neural network module
            loss (nn.loss): loss function
            opt (torch.optim): optimizer

        Returns:
            loss
        """
        super().__init__()
        self.G_x = G_x # load pre-trained model
        self.model = f_c(input_shape=input_shape, concept_shape=concept_shape, output_shape=output_shape)
        self.loss_classify = loss_fn_classify
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.0001)

    def forward(self, x):
        # apply model layers
        self.G_x.eval()
        x = self.G_x(x)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx, loss_name = 'train_loss'):
        # training
        x, c1, c2, y = batch
        z = self.forward(x)
        loss_prediction = self.loss_classify(z, y)
        loss = loss_prediction 

        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        # validation
        x, c1, c2, y = batch
        z = self.forward(x)
        loss_prediction = self.loss_classify(z, y)
        loss = loss_prediction 


        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def test_step(self, batch, batch_idx, loss_name ='test_loss'):
        #test
        x, c1, c2, y = batch
        z = self.forward(x)
        loss_prediction = self.loss_classify(z, y)
        loss = loss_prediction 

        # calculate acc

        # # std. label
        _, predicted = torch.max(z.data, 1) 
        test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

        self.log_dict({'test_loss': loss, 'test_acc': test_acc})
        return loss

    def configure_optimizers(self):
        return self.optimiser
    