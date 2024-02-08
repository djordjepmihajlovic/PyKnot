# torch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# lightning modules
import pytorch_lightning as pl

import seaborn as sns
import matplotlib.pyplot as plt

################## <--FFNN--> ###################

class FFNNModel(nn.Module):
    # define model structure i.e. layers
    def __init__(self, input_shape, output_shape, norm, predict):
        """nn.Module initialization --> builds neural network

        Args:
            input_shape (list): dim of input array
            output_shape (int): size of output array
            norm (bool): norm

        Returns:
            nn.Module
        """
        super(FFNNModel, self).__init__()
        self.flatten_layer = nn.Flatten()
        self.pred = predict
        
        # init layers
        if norm:
            self.bn_layer = nn.BatchNorm1d(input_shape[0])
            self.dense_layer1 = nn.Linear(input_shape[0]*input_shape[1], 320)
        else:
            self.dense_layer1 = nn.Linear(input_shape[0]*input_shape[1], 320)

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


        if self.pred == "class":
            return F.softmax(x, dim=1) 
        
        elif self.pred == "dowker":
            return x.view(-1, 7, 1) # <- have: StA_2_DT (-1, 32, 1) (32 is for generated dowker code)
        
        elif self.pred == "jones":
            return x.view(-1, 10, 2) # <- have: polynomial (power, factor) [one hot encoding] nb. 3_1: q^(-1)+q^(-3)-q^(-4) = [1, 0, 1, 1][1, 0, 1, -1]
        
        # also technically this makes little sense, i think the neural network is just 'learning' the different knot types
        # and then choosing the correct label :(
        
        elif self.pred == "quantumA2":
            return x.view(-1, 31, 2) # <- same as Jones
        


################## <--RNN--> ###################

class RNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, norm):
        super(RNNModel, self).__init__()

        # if norm:
        #     self.bn_layer = nn.BatchNorm1d(input_shape[0])
        #     self.lstm1 = nn.LSTM(input_shape[0]*input_shape[1], 100)
        # else:
        #     self.lstm1 = nn.LSTM(input_shape[0]*input_shape[1], 100, batch_first=True, bidirectional=False)

        # self.lstm2 = nn.LSTM(100, 100, batch_first=True, bidirectional=True)
        # self.lstm3 = nn.LSTM(100 * 2, 100, batch_first=True, bidirectional=False)

        self.lstm1 = nn.LSTM(input_shape[1], 64, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(100, output_shape)

    def forward(self, x):

        self.lstm1.flatten_parameters()

        out, _ = self.lstm1(x)
        out = F.tanh(out)
        
        out, _ = self.lstm2(out)
        out = F.tanh(out[:, -1, :])  # taking output from the last time step
        
        out = self.fc(out)

        # return out # <- std
        if self.pred == "class":
            return F.softmax(x, dim=1) 
        
        elif self.pred == "dowker":
            return x.view(-1, 7, 1) # <- have: StA_2_DT (-1, 32, 1) (32 is for generated dowker code)
        
        elif self.pred == "jones":
            return x.view(-1, 10, 2) # <- have: polynomial (power, factor) [one hot encoding] nb. 3_1: q^(-1)+q^(-3)-q^(-4) = [1, 0, 1, 1][1, 0, 1, -1]
        
        # also technically this makes little sense, i think the neural network is just 'learning' the different knot types
        # and then choosing the correct label :(
        
        elif self.pred == "quantumA2":
            return x.view(-1, 31, 2) # <- same as Jones

################## <--CNN--> ###################

class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, norm):
        super(CNNModel, self).__init__()

        # init layers
        self.norm = norm
        if self.norm:
            self.bn_layer = nn.BatchNorm2d(input_shape[0])

        # hidden layers to CNN
        # in -> out -> kernel 
        self.conv_layer1 = nn.Conv1d(input_shape[0]*input_shape[1], 32, kernel_size = 3, stride = 1, padding = 1)
        # takes in input shape (100) -> outputs shape (32) with a 2D kernel size? (3, 3)
        # self.max_pool_layer1 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.conv_layer2 = nn.Conv2d(256, 20, kernel_size = 3, stride=1, padding=1)
        self.flatten_layer = nn.Flatten()
        self.dense_layer = nn.Linear(32, 80)

        # output layer
        self.output_layer = nn.Linear(80, output_shape)

    def forward(self, x):

        # if self.mask_value is not None:
        #     x = x * self.g
        if self.norm:
            x = self.bn_layer(x)

        x = F.relu(self.conv_layer1(x))
        # x = self.max_pool_layer1(x)
        x = F.relu(self.conv_layer2(x))
        x = self.flatten_layer(x)
        x = F.relu(self.dense_layer(x))
        x = self.output_layer(x)
        x = F.softmax(self.output_layer(x), dim=1)

        return x
    

################## <--Graph Neural Network--> ###################   
    
    


################## <--General Neural Network, Pytorch LightningModule for train/val/test config--> ###################
    
class NN(pl.LightningModule):
    def __init__(self, model, loss, opt):
        """pl.LightningModule --> builds train/val/test 

        Args:
            model (nn.Module): neural network module
            loss (nn.loss): loss function
            opt (torch.optim): optimizer

        Returns:
            loss
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimiser = opt

    def forward(self, x):
        # apply model layers
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx, loss_name = 'train_loss'):
        # training
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        # validation
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y) 
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    
    def test_step(self, batch, batch_idx, loss_name ='test_loss'):
        #test
        x, y = batch 
        z = self.forward(x)
        loss = self.loss(z, y) 

        # calculate acc

        # # std. label
        # _, predicted = torch.max(z.data, 1) 
        # test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

        ## dowker
        true = 0
        false = 0
        predicted = torch.round(z)
        el = (y-predicted)

        for idx, i in enumerate(el):
            if torch.sum(i) == 0.0:
                true += 1
            else:
                false += 1
                # print(f"true: {y[idx]}")
                # print(f"predicted: {predicted[idx]}")

        test_acc = true/(true+false)
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
    
################## <--FFNN config--> ###################

def setup_FFNN(input_shape, output_shape, opt, norm, loss, predict):
    """setup function --> defines required network using helper

    Args:
        input_shape (list): dim of input array
        output_shape (int): size of output array
        opt (str): required optimizer (adam, sgd)
        norm (bool): norm
        loss (str): required loss function (CEL, MSE)

    Returns:
        model (nn.Module)
        loss (nn.loss)
        optimizer (torch.optim)
    """

    # model
    model = FFNNModel(input_shape, output_shape, norm, predict)

    # loss function 
    if loss == "CEL":
        loss_fn = nn.CrossEntropyLoss() 
    elif loss == "MSE":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Invalid loss choice; 'CEL' or 'MSE'.")

    # optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.000001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice; 'adam' or 'sgd'.")

    return model, loss_fn, optimizer

################## <--RNN config--> ###################

def setup_RNN(input_shape, output_shape, opt, norm, loss):
    """setup function --> defines required network using helper

    Args:
        input_shape (list): dim of input array
        output_shape (int): size of output array
        opt (str): required optimizer (adam, sgd)
        norm (bool): norm
        loss (str): required loss function (CEL, MSE)

    Returns:
        model (nn.Module)
        loss (nn.loss)
        optimizer (torch.optim)
    """

    # model
    model = RNNModel(input_shape, output_shape, norm)

    # loss function 
    if loss == "CEL":
        loss_fn = nn.CrossEntropyLoss() 
    elif loss == "MSE":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Invalid loss choice; 'CEL' or 'MSE'.")

    # optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice; 'adam' or 'sgd'.")

    return model, loss_fn, optimizer

################## <--CNN config--> ###################

def setup_CNN(input_shape, output_shape, opt, norm, loss):
    """setup function --> defines required network using helper

    Args:
        input_shape (list): dim of input array
        output_shape (int): size of output array
        opt (str): required optimizer (adam, sgd)
        norm (bool): norm
        loss (str): required loss function (CEL, MSE)

    Returns:
        model (nn.Module)
        loss (nn.loss)
        optimizer (torch.optim)
    """

    # model
    model = CNNModel(input_shape, output_shape, norm)

    # loss function 
    if loss == "CEL":
        loss_fn = nn.CrossEntropyLoss() 
    elif loss == "MSE":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Invalid loss choice; 'CEL' or 'MSE'.")

    # optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice; 'adam' or 'sgd'.")

    return model, loss_fn, optimizer


