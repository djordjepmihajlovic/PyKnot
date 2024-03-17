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
            predict (str): prediction type

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
        
        elif self.pred == "v2" or self.pred == "v3":
            return x.view(-1, 1)
        
        elif self.pred == "dowker":
            return x.view(-1, 32, 1) 
        
        elif self.pred == "jones":
            return x.view(-1, 10, 2) # <- have: polynomial (power, factor) [one hot encoding] nb. 3_1: q^(-1)+q^(-3)-q^(-4) = [1, 0, 1, 1][1, 0, 1, -1]
        
        # also technically this makes little sense, i think the neural network is just 'learning' the different knot types
        # and then choosing the correct label :(
        
        elif self.pred == "quantumA2":
            return x.view(-1, 31, 2) # <- same as Jones

################## <--FFNN to StS combinations--> ###################

class FFNN_Combinatoric(nn.Module):
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
        super(FFNN_Combinatoric, self).__init__()
        self.flatten_layer = nn.Flatten()
        self.pred = predict
        
        # init layers
        if norm:
            self.bn_layer = nn.BatchNorm1d(input_shape[0])
            self.combinatoric_layer1 = nn.Linear(input_shape[0]*input_shape[1], input_shape[0]*input_shape[1])
        else:
            self.combinatoric_layer1 = nn.Linear(input_shape[0]*input_shape[1], input_shape[0]*input_shape[1])

        # hidden layers to FFNN
        # self.combinatoric_layer2 = nn.Linear((input_shape[0]*input_shape[1]), input_shape[0]*input_shape[1])
        # self.combinatoric_layer3 = nn.Linear((input_shape[0]*input_shape[1]), (input_shape[0]*input_shape[1]))

        # output layer
        self.output_layer = nn.Linear((input_shape[0]*input_shape[1]), output_shape)

    def forward(self, x):
        
        x = self.flatten_layer(x)

        if hasattr(self, 'bn_layer'):
            x = self.bn_layer(x)

        # separate layers -> hopefully this will allow the network to learn the different combinations
        x_ij = self.combinatoric_layer1(x)
        # x_kl = self.combinatoric_layer2(x)
        # x_mn = self.combinatoric_layer3(x)

        # intermediate multiplied layers
        # x_int = torch.mul(x_ij, x_kl)
        # x_int = torch.mul(x_int, x_mn)

        x = self.output_layer(x_ij)

        if self.pred == "class" or "latent":
            return F.softmax(x, dim=1) 


################## <--RNN--> ###################

class RNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, norm, predict):
        super(RNNModel, self).__init__()

        self.hidden_size = 128
        self.num_layers = 2
        self.seq = self.inp = input_shape[0]

        self.lstm = nn.LSTM(input_shape[1], self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_shape)
        self.pred = predict

    def forward(self, x):

        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # nb. x.size(0) is the feature per timestep
        cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (hidden, cell)) 
        out = self.fc(out[:, -1, :])  # take output from last time step for all sequences 

        # return out # <- std
        if self.pred == "class":
            return F.softmax(out, dim=1) 
        
        elif self.pred == "v2" or self.pred == "v3":
            return out.view(-1, 1)
        
        elif self.pred == "dowker":
            return out.view(-1, 32, 1) 
        
        elif self.pred == "jones":
            return out.view(-1, 10, 2) # <- have: polynomial (power, factor) [one hot encoding] nb. 3_1: q^(-1)+q^(-3)-q^(-4) = [1, 0, 1, 1][1, 0, 1, -1]
        
        elif self.pred == "quantumA2":
            return out.view(-1, 31, 2) 

################## <--CNN--> ###################

class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, norm, predict):
        super(CNNModel, self).__init__()

        #nb. input_shape = (256, 100, 100)
        self.pred = predict

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(32 * 25 * 25, 128)  
        self.fc2 = nn.Linear(128, output_shape)  

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))  # [256, 16, 100, 100] -> [256, 16, 50, 50]
        x = self.pool(F.relu(self.conv2(x)))  # [256, 32, 50, 50] -> [256, 32, 25, 25]
        
        # Flatten the feature maps
        x = x.view(-1, 32 * 25 * 25)  # Reshape to [256, 32 * 25 * 25]
        
        # Fully connected layers with activation
        x = F.relu(self.fc1(x))  # [256, 32 * 25 * 25] -> [256, 128]
        x = self.fc2(x)  # [256, 128] -> [256, 1]
        
        # return out # <- std
        if self.pred == "class":
            return F.softmax(x, dim=1) 
        
        elif self.pred == "v2" or self.pred == "v3":
            return x.view(-1, 1)
        
        elif self.pred == "dowker":
            return x.view(-1, 32, 1) # 
        
        elif self.pred == "jones":
            return x.view(-1, 10, 2) # <- have: polynomial (power, factor) [one hot encoding] nb. 3_1: q^(-1)+q^(-3)-q^(-4) = [1, 0, 1, 1][1, 0, 1, -1]
        
        elif self.pred == "quantumA2":
            return x.view(-1, 31, 2) # <- same as Jones
    

################## <--Graph Neural Network--> ###################   
    
# wouldve loved to implement... would be extremely interesting for studying knots... future work perhaps

################## <--General Neural Network, Pytorch LightningModule for train/val/test config--> ###################
    
class NN(pl.LightningModule):
    def __init__(self, model, loss, opt, predict):
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
        self.predict = predict

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

        if self.predict == "class":
        # std. label
            _, predicted = torch.max(z.data, 1) 
            test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

        else:
        # ## invariant
            true = 0
            false = 0
            el = (y-z)

            true50 = 0
            false50 = 0

            for idx, i in enumerate(el):
                if i < 0.1:
                    true += 1
                else:
                    false += 1
                    # print(f"true: {y[idx]}")
                    # print(f"predicted: {predicted[idx]}")

                if i < 0.5:
                    true50 += 1
                else:
                    false50 += 1

            test_acc = true / (true + false)

        # log outputs
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})

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
       
    if predict == "combinatoric":
        model = FFNN_Combinatoric(input_shape, output_shape, norm, predict)
    else:
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
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice; 'adam' or 'sgd'.")

    return model, loss_fn, optimizer

################## <--RNN config--> ###################

def setup_RNN(input_shape, output_shape, opt, norm, loss, predict):
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
    model = RNNModel(input_shape, output_shape, norm, predict)

    # loss function 
    if loss == "CEL":
        loss_fn = nn.CrossEntropyLoss() 
    elif loss == "MSE":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Invalid loss choice; 'CEL' or 'MSE'.")

    # optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice; 'adam' or 'sgd'.")
    
    print(model)

    return model, loss_fn, optimizer

################## <--CNN config--> ###################

def setup_CNN(input_shape, output_shape, opt, norm, loss, predict):
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
    model = CNNModel(input_shape, output_shape, norm, predict)

    # loss function 
    if loss == "CEL":
        loss_fn = nn.CrossEntropyLoss() 
    elif loss == "MSE":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Invalid loss choice; 'CEL' or 'MSE'.")

    # optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice; 'adam' or 'sgd'.")

    return model, loss_fn, optimizer


