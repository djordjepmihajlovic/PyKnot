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
    
################## <--xyz FFNN Encoders--> ###################

class Encoder_XYZ(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder_XYZ, self).__init__()
        self.flatten_layer = nn.Flatten()

        self.linear1x = nn.Linear(input_shape[0]*input_shape[1], 320) # here is size of input data x1 (dimension flattened) and following hidden layer x2
        self.linear2x = nn.Linear(320, 120)
        self.linear3x = nn.Linear(120, 64)
        self.linear4x = nn.Linear(64, latent_dims)

        # self.linear1y = nn.Linear(100, 320) 
        # self.linear2y = nn.Linear(320, 120)
        # self.linear3y = nn.Linear(120, 64)
        # self.linear4y = nn.Linear(64, latent_dims)

        # self.linear1z = nn.Linear(100, 320) 
        # self.linear2z = nn.Linear(320, 120)
        # self.linear3z = nn.Linear(120, 64)
        # self.linear4z = nn.Linear(64, latent_dims)

    def forward(self, x):

        x = self.flatten_layer(x)
        x = F.leaky_relu(self.linear1x(x))
        x = F.leaky_relu(self.linear2x(x))
        x = F.leaky_relu(self.linear3x(x))

        # y = self.flatten_layer(y)  
        # y = F.leaky_relu(self.linear1y(y))
        # y = F.leaky_relu(self.linear2y(y))
        # y = F.leaky_relu(self.linear3y(y))

        # z = self.flatten_layer(z)
        # z = F.leaky_relu(self.linear1z(z))
        # z = F.leaky_relu(self.linear2z(z))
        # z = F.leaky_relu(self.linear3z(z))

        return self.linear4x(x)
    
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
    
################## <--RNN, XYZ Encoder--> ###################
    
class Encoder_RNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder_RNN, self).__init__()

        self.lstm1 = nn.LSTM(input_shape[1], 64, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        # self.lstm3 = nn.LSTM(100 * 2, 100, batch_first=False, bidirectional=False)
        self.fcmu = nn.Linear(64, latent_dims)
    
    def forward(self, x):
        # Flatten parameters for LSTM layers
        self.lstm1.flatten_parameters()
        # self.lstm2.flatten_parameters()
        # self.lstm3.flatten_parameters()

        x, _ = self.lstm1(x)
        x = torch.tanh(x)
        x, _ = self.lstm2(x)
        x = torch.tanh(x[:, -1, :])  # taking output from the last time step
    
        mu = self.fcmu(x)

        return mu
    
################## <--CNN Encoder--> ###################
    
class Encoder_CNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder_CNN, self).__init__()

        self.flatten_layer = nn.Flatten(start_dim=1)

        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.linear_mu = nn.Linear(64 * (input_shape[0]), latent_dims)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = F.relu(x)   
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten_layer(x)
        mu = self.linear_mu(x)

        return mu

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
    
################## <--StS Decoder--> ###################
    
class Decoder_StS(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder_StS, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 320)
        self.linear2 = nn.Linear(320, 100*100) #input_shape[0]*input_shape[1])

    def forward(self, z):
        z = F.leaky_relu(self.linear1(z))
        z = self.linear2(z) 

        return z.reshape((-1, 100, 100)) 
    
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
    
################## <--CNN, XYZ Decoder--> ###################
    
class Decoder_CNN(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder_CNN, self).__init__()

        self.input_shape = input_shape

        self.linear1 = nn.Linear(latent_dims, 64 * (input_shape[0]))
        self.conv1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=input_shape[1], kernel_size=3, padding=1)

    def forward(self, z):

        z = self.linear1(z)
        z = z.view(-1, 64, (self.input_shape[0]))   
        z = self.conv1(z)
        z = F.relu(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = z.permute(0, 2, 1)

        return z
    
################## <--Decoder Cond--> ###################
    
class Decoder_StA_Cond(nn.Module):
    def __init__(self, input_shape, cond_shape, latent_dims):
        super(Decoder_StA_Cond, self).__init__()

        self.linear1 = nn.Linear(latent_dims + cond_shape[0]*cond_shape[1], 320)
        self.linear2 = nn.Linear(320, input_shape[0]*input_shape[1])

    def forward(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        z = F.leaky_relu(self.linear1(z))
        z = self.linear2(z) 

        return z.reshape((-1, 100, 3)) 
    
################## <--Autoencoder (pl.LightningModule), forward (enc+dec), train, test, val--> ###################
    
class Autoencoder(pl.LightningModule):

    def __init__(self,  input_shape, latent_dims,  loss, opt):
        super().__init__()

        self.loss_fn = loss
        self.optimiser = opt
        self.encoder = Encoder_XYZ(input_shape = input_shape, latent_dims = latent_dims)
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

        return mu, log_sigma 
    
################## <--Cond .VAE Encoder (two encodings= mean encoding and deviation encoding)--> ###################

class Cond_VariationalEncoder(nn.Module):
    def __init__(self, input_shape, cond_shape, latent_dims):
        super(Cond_VariationalEncoder, self).__init__()
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(input_shape[0]*input_shape[1] + cond_shape[0]*cond_shape[1], 320)
        self.linear2 = nn.Linear(320, 64) 
        self.linear3 = nn.Linear(64, latent_dims)
        self.linear4 = nn.Linear(64, latent_dims)

    def forward(self, x, cond):
        # x = torch.flatten(x, start_dim=1)
        x = self.flatten(x)
        x = torch.cat([x, cond], dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        mu = self.linear3(x) # mean (mu) layer
        log_sigma = self.linear4(x) # log variance layer

        return mu, log_sigma 
    
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
        self.decoder = Decoder_StA(input_shape = input_shape, latent_dims = latent_dims)

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

################## <--VariationalAutoencoder (pl.LightningModule), forward (enc+dec), train, test, val--> ###################
    
class ConditionalVAE(pl.LightningModule):

    def __init__(self,  input_shape, cond_shape, latent_dims,  loss, opt, beta):

        super().__init__()

        self.optimiser = opt
        self.beta = beta
        self.encoder = Cond_VariationalEncoder(input_shape = input_shape, cond_shape = cond_shape, latent_dims = latent_dims)
        self.decoder = Decoder_StA_Cond(input_shape = input_shape, cond_shape = cond_shape, latent_dims = latent_dims)

    def reparameterization(self, mu, sigma):

        epsilon = torch.randn_like(sigma)                    
        z = mu + sigma*epsilon                          
        return z
        
    def forward(self, x, cond):

        mean, log_sigma = self.encoder(x, cond)
        z = self.reparameterization(mean, torch.exp(0.5 * log_sigma)) 
        x_hat = self.decoder(z, cond)
        return x_hat, mean, log_sigma, z

    def configure_optimizers(self):

        if self.optimiser == 'adam':
            return torch.optim.Adam(self.parameters(), lr=0.0001) # 0.001 FFNN 
        
    def loss_function(self, x, x_hat, mean, log_sigma):

        MSE = F.mse_loss(x_hat, x, reduction='sum') # reduction sum returned best results for StA
        KLD = self.beta*(- 0.5 * torch.sum(1+ log_sigma - mean.pow(2) - log_sigma.exp()))

        return MSE + KLD

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, cond, y = batch
        x_hat, mean, log_sigma = self.forward(x, cond)

        loss = self.loss_function(x, x_hat, mean, log_sigma) # x in place of y usually
        self.log(loss_name, loss, on_epoch=True, on_step=True)

        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, cond, y = batch
        x_hat, mean, log_sigma = self.forward(x, cond)

        loss = self.loss_function(x, x_hat, mean, log_sigma) # x in place of y usually
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, cond, y = batch 
        x_hat, mean, log_sigma = self.forward(x, cond)

