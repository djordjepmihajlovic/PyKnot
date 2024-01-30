# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# lightning modules
import pytorch_lightning as pl

################## <--FFNN Encoder--> ###################

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
    
################## <--FFNN Encoder with Attention--> ###################

class StSEncoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(StSEncoder, self).__init__()
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
    
################## <--RNN Encoder--> ###################
    
class Encoder_RNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder_RNN, self).__init__()

        self.lstm1 = nn.LSTM(input_shape[0]*input_shape[1], 100, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(100, 100, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(100 * 2, 100, batch_first=True, bidirectional=False)
        self.fcmu = nn.Linear(100, latent_dims)
    
    def forward(self, x):
        # Flatten parameters for LSTM layers
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()

        x, _ = self.lstm1(x)
        x = torch.tanh(x)

        x, _ = self.lstm2(x)
        x = torch.tanh(x)

        x, _ = self.lstm3(x)
        x = torch.tanh(x[:, -1, :])  # taking output from the last time step
    
        mu = self.fcmu(x)

        return mu
    
################## <--CNN Encoder--> ###################
    
class Encoder_CNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder_CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc1 = nn.Linear(1024 * 3 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool2(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, 1024 * 3 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


################## <--Attention using StS data--> ###################

class Attention(nn.Module):
    def __init__(self, input_shape):
        super(Attention, self).__init__()
        
        self.flatten_layer = nn.Flatten()
        self.linear = nn.Linear(input_shape[0]*input_shape[1], 100)

    def forward(self, x, weights):

        x = self.flatten_layer(x)
        weights = self.flatten_layer(weights)

        attention_weight = F.softmax(weights) 
        attention_weight = self.flatten_layer(attention_weight)

        # need to figure out how to multiply this properly

        print(x.size())
        print(attention_weight.size()) 

        print(x[0])

        print(attention_weight[0])

        output = attention_weight*x

        print(output.size())

        return output # this should be a 100x100 tensor (check)
    
################## <--StA Decoder--> ###################
    
class Decoder_StA(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder_StA, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 320)
        self.linear2 = nn.Linear(320, 100) #input_shape[0]*input_shape[1])

    def forward(self, z):
        z = F.leaky_relu(self.linear1(z))
        z = self.linear2(z) 

        return z.reshape((-1, 100, 1)) 
    
################## <--XYZ Decoder-> ###################
    
class Decoder_XYZ(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder_XYZ, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 320)
        self.linear2 = nn.Linear(320, input_shape[0]*input_shape[1])

    def forward(self, z):
        z = F.leaky_relu(self.linear1(z))
        z = self.linear2(z) 

        return z.reshape((-1, 100, 3)) 
    
################## <--Autoencoder (pl.LightningModule), forward (enc+dec), train, test, val--> ###################
    
class Autoencoder(pl.LightningModule):

    def __init__(self,  input_shape, latent_dims,  loss, opt):
        super().__init__()

        self.loss_fn = loss
        self.optimiser = opt
        self.encoder = Encoder_RNN(input_shape = input_shape, latent_dims = latent_dims)
        self.decoder = Decoder_XYZ(input_shape = input_shape, latent_dims = latent_dims)
        
    def forward(self, x):

        # apply encoder
        x = self.encoder(x)
        # apply decoder
        x = self.decoder(x)

        return x
    
    def configure_optimizers(self):

        if self.optimiser == 'adam':
            return torch.optim.Adam(self.parameters(), lr=0.0001)

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
    
################## <--Autoencoder with Attention (pl.LightningModule), forward (enc+dec), train, test, val--> ###################

class AttentionAutoencoder(pl.LightningModule):

    def __init__(self,  input_shape, output_shape, latent_dims,  loss, opt):
        super().__init__()

        self.loss_fn = loss
        self.optimiser = opt
        self.encoder = StSEncoder(input_shape = input_shape, latent_dims = latent_dims)
        self.decoder = Decoder_XYZ(output_shape = output_shape, latent_dims = latent_dims)
        
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
        data_StS, data_XYZ, label = batch
        reconstruction = self.forward(data_StS)

        loss = self.loss_fn(reconstruction, data_XYZ) 
        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        data_StS, data_XYZ, label = batch
        reconstruction = self.forward(data_StS)
        loss = self.loss_fn(reconstruction, data_XYZ) 
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        data_StS, data_XYZ, label = batch 
        reconstruction = self.forward(data_StS)

################## <--VAE Encoder (two encodings= mean encoding and deviation encoding)--> ###################

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

        return mu, log_sigma  # not currently log_sigma has a tanh (for XYZ)
    
################## <--VAE RNN Encoder (two encodings= mean encoding and deviation encoding)--> ###################
    
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
    
################## <--VariationalAutoencoder (pl.LightningModule), forward (enc+dec), train, test, val--> ###################
    
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

        MSE = F.mse_loss(x_hat, x, reduction='sum') # reduction sum returned best results for StA
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

