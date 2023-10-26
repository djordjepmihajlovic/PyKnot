# pytorch --> framework
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# lightning --> train/val/test initialization
import pytorch_lightning as pl

class NNModel(nn.Module):
    # define model structure i.e. layers
    def __init__(self, input_shape, output_shape, norm):
        """nn.Module initialization --> builds neural network

        Args:
            input_shape (list): dim of input array
            output_shape (int): size of output array
            norm (bool): norm

        Returns:
            nn.Module
        """
        super(NNModel, self).__init__()
        self.flatten_layer = nn.Flatten()
        
        # init layers
        if norm:
            self.bn_layer = nn.BatchNorm1d(input_shape[0])
            self.dense_layer1 = nn.Linear(input_shape[0]*input_shape[1], 320)
        else:
            self.dense_layer1 = nn.Linear(input_shape[0]*input_shape[1], 320)

        # hidden layers to NN
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
        # x = F.sigmoid(x)
        # return x.view(-1, 100, 3)
    


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
        _, predicted = torch.max(z.data, 1)
        test_acc = torch.sum(y == predicted).item() / (len(y)*1.0)

        # log outputs
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})
        # self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return self.optimiser

    

def setup_NN(input_shape, output_shape, opt, norm, loss):
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
    model = NNModel(input_shape, output_shape, norm)

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


### -- check documentation for metrics callback -- ###

# class MetricsCallback(Callback):

#      def __init__(self):
#           super().__init__()
#           self.metrics = {"train_loss":[], "val_loss":[]}

#      def on_validation_end(self, trainer, pl_module):
#           if "train_loss" in trainer.callback_metrics.keys(): 
#                self.metrics["train_loss"].append(copy.deepcopy(trainer.callback_metrics["train_loss"]).np())
#           if "val_loss" in trainer.callback_metrics.keys(): 
#                self.metrics["val_loss"].append(copy.deepcopy(trainer.callback_metrics["val_loss"]).np())
         
#      def on_train_batch_end(self, trainer, pl_module, outputs,batch,batch_idx):
#           if "train_loss" in trainer.callback_metrics.keys(): 
#                self.metrics["train_loss"].append(copy.deepcopy(trainer.callback_metrics["train_loss"]).np())
#           if "val_loss" in trainer.callback_metrics.keys(): 
#                self.metrics["val_loss"].append(copy.deepcopy(trainer.callback_metrics["val_loss"]).np())
         
