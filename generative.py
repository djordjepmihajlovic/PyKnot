# idea with variational autoencoder - theory based on Stanford CS229: VAE's

# below is just GENERIC code for formatting of an autoencoder and variational autoencoder (will probably require some 
# reformatting for our data)
# this would then be implemented into training function on main.py similarly to any nn.module

# autoencoder
# manifold hypothesis -> high-dim data consists of low-dim data embedded in that high-dim space
# PCA & t-SNE are dim-reductionality techniques
# autoencoders are type of nn used to perform dim-reductionality
# encoder -> decoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder, self).__init__()
        self.flatten_layer = nn.Flatten()

        self.linear1 = nn.Linear(input_shape[0]*input_shape[1], 320) # here is size of input data x1 (dimension flattened) and following hidden layer x2
        self.linear2 = nn.Linear(320, latent_dims)

    def forward(self, x):
        x = self.flatten_layer(x)

        x = F.relu(self.linear1(x))
        return self.linear2(x)
    
class Decoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 320)
        self.linear2 = nn.Linear(320, input_shape[0]*input_shape[1])

    def forward(self, z):
        z = F.relu(self.linear1(z))

        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 100, 1)) # here x3 is output shape of data, in case of 100 beads XYZ that would be (-1, 1, 100, 3) I think
                                          # (-1, 1, 100, 1) for SIGWRITHE -> generating SIGWRITHE data
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(x)
    

# VAE
# how to solve a disjointed and non-continuous latent space?
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(x1, x2)
        self.linear2 = nn.Linear(x2, latent_dims)
        self.linear3 = nn.Linar(x2, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.k1 = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.k1 = (sigma**2 + mu**2 -torch.log(sigma) -1/2).sum()

        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    