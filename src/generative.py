# idea with variational autoencoder - theory based on Stanford CS229: VAE's

# below is just GENERIC code for formatting of an autoencoder and variational autoencoder (will probably require some 
# reformatting for our data)
# this would then be implemented into training function on main.py similarly to any nn.module

# autoencoder
# manifold hypothesis -> high-dim data consists of low-dim data embedded in that high-dim space
# PCA & t-SNE are dim-reductionality techniques
# autoencoders are type of nn used to perform dim-reductionality
# encoder -> decoder

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

#lightning
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder, self).__init__()
        self.flatten_layer = nn.Flatten()

        self.linear1 = nn.Linear(input_shape[0]*input_shape[1], 320) # here is size of input data x1 (dimension flattened) and following hidden layer x2
        self.linear2 = nn.Linear(320, 120)
        self.linear3 = nn.Linear(120, 64)
        self.linear4 = nn.Linear(64, latent_dims)

    def forward(self, x):
        x = self.flatten_layer(x)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return self.linear4(x)
    
class Encoder_RNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder_RNN, self).__init__()

        self.lstm1 = nn.LSTM(input_shape[0]*input_shape[1], 100, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(100, 100, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(100 * 2, 100, batch_first=True, bidirectional=False)
        self.fcmu = nn.Linear(100, latent_dims)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = F.tanh(x)
        x, _ = self.lstm2(x)
        x = F.tanh(x)
        x, _ = self.lstm3(x)
        x = F.tanh(x[:, -1, :])  # taking output from the last time step
    
        mu = self.fcmu(x)

        return mu
    
class Decoder_StA(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder_StA, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 320)
        self.linear2 = nn.Linear(320, 100) #input_shape[0]*input_shape[1])

    def forward(self, z):
        z = F.leaky_relu(self.linear1(z))

        # z = torch.sigmoid(self.linear2(z))
        z = self.linear2(z) # no sigmoid
        return z.reshape((-1, 100, 1)) 
    
class Decoder_XYZ(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder_XYZ, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 320)
        self.linear2 = nn.Linear(320, input_shape[0]*input_shape[1])

    def forward(self, z):
        z = F.leaky_relu(self.linear1(z))

        # z = torch.sigmoid(self.linear2(z))
        z = self.linear2(z) # no sigmoid
        return z.reshape((-1, 100, 3)) 
    
class Autoencoder(pl.LightningModule):

    def __init__(self,  input_shape, latent_dims,  loss, opt):
        # stats holds args for encoder and decoder
        super().__init__()

        self.loss_fn = loss
        self.optimiser = opt
        self.encoder = Encoder(input_shape = input_shape, latent_dims = latent_dims)
        self.decoder = Decoder_XYZ(input_shape = input_shape, latent_dims = latent_dims)
        
    def forward(self, x):

        # apply encoder
        x = self.encoder(x)
        # apply decoder
        x = self.decoder(x)

        return x
    
    def configure_optimizers(self):

        if self.optimiser == 'adam':
            return torch.optim.Adam(self.parameters(), lr=0.00001)

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, y = batch
        z = self.forward(x)

        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch 
        z = self.forward(x)
    

# VAE
# how to solve a disjointed and non-continuous latent space?
class VariationalEncoderFFNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(VariationalEncoderFFNN, self).__init__()
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(input_shape[0]*input_shape[1], 320)
        self.linear2 = nn.Linear(320, 64) 
        self.linear3 = nn.Linear(64, latent_dims)
        self.linear4 = nn.Linear(64, latent_dims)

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        x = self.flatten(x)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        mu = self.linear3(x) # mean (mu) layer
        log_sigma = self.linear4(x) # log variance layer

        return mu, F.tanh(log_sigma)  # not currently log_sigma has a tanh (for XYZ)
    
class VariationalEncoderRNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(VariationalEncoderRNN, self).__init__()

        self.lstm1 = nn.LSTM(input_shape[0]*input_shape[1], 100, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(100, 100, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(100 * 2, 100, batch_first=True, bidirectional=False)
        self.fcmu = nn.Linear(100, latent_dims)
        self.fcsig = nn.Linear(100, latent_dims)

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = F.tanh(x)
        x, _ = self.lstm2(x)
        x = F.tanh(x)
        x, _ = self.lstm3(x)
        x = F.tanh(x[:, -1, :])  # taking output from the last time step
    
        mu = self.fcmu(x)
        log_sigma = self.fcsig(x)

        return mu, log_sigma
    
class VariationalAutoencoder(pl.LightningModule):

    def __init__(self,  input_shape, latent_dims,  loss, opt, beta):

        super().__init__()

        self.optimiser = opt
        self.beta = beta
        self.encoder = VariationalEncoderFFNN(input_shape = input_shape, latent_dims = latent_dims)
        self.decoder = Decoder_XYZ(input_shape = input_shape, latent_dims = latent_dims)

    def reparameterization(self, mu, sigma):

        epsilon = torch.randn_like(sigma)                    
        z = mu + sigma*epsilon                          
        return z
        
    def forward(self, x):

        mean, log_sigma = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_sigma)) 
        x_hat = self.decoder(z)
        return x_hat, mean, log_sigma

    def configure_optimizers(self):

        if self.optimiser == 'adam':
            return torch.optim.Adam(self.parameters(), lr=0.001) # 0.001 FFNN 
        
    def loss_function(self, x, x_hat, mean, log_sigma):

        MSE = F.mse_loss(x_hat, x, reduction='mean') # reduction sum returned best results for StA
        KLD = self.beta*(- 0.5 * torch.sum(1+ log_sigma - mean.pow(2) - log_sigma.exp()))

        return MSE + KLD

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, y = batch
        x_hat, mean, log_sigma = self.forward(x)

        loss = self.loss_function(x, x_hat, mean, log_sigma) # x in place of y usually
        self.log(loss_name, loss, on_epoch=True, on_step=True)

        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        x_hat, mean, log_sigma = self.forward(x)

        loss = self.loss_function(x, x_hat, mean, log_sigma) # x in place of y usually
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch 
        z = self.forward(x)

