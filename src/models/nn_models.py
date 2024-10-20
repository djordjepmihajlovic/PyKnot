# torch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# lightning modules
import pytorch_lightning as pl

import csv

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
        
        # i think the neural network is just 'learning' the different knot types
        # and then choosing the correct label :(
        
        elif self.pred == "quantumA2":
            return x.view(-1, 31, 2) # <- same as Jones

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
        
        elif self.pred == "v2v3":
            return out.view(-1, 2, 1)
        
        elif self.pred == "dowker":
            return out.view(-1, 32, 1) 
        
        elif self.pred == "jones":
            return out.view(-1, 10, 2) # <- have: polynomial (power, factor) [one hot encoding] nb. 3_1: q^(-1)+q^(-3)-q^(-4) = [1, 0, 1, 1][1, 0, 1, -1]

################## <--CNN--> ###################

class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, norm, predict):
        super(CNNModel, self).__init__()

        #nb. input_shape = (256, 100, 100)
        self.pred = predict

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=5, padding=0)
        
        self.fc1 = nn.Linear(128 * 5 * 5, 128)  
        self.fc2 = nn.Linear(128, output_shape)  

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))  # [256, 16, 100, 100] -> [256, 16, 50, 50]
        x = self.pool(F.relu(self.conv2(x)))  # [256, 32, 50, 50] -> [256, 32, 25, 25]
        x = self.pool(F.relu(self.conv3(x)))  # [256, 64, 25, 25] -> [256, 64, 5, 5]

        
        # Flatten the feature maps
        x = x.view(-1, 128 * 5 * 5)  # Reshape to [256, 128 * 5 * 5]
        
        # Fully connected layers with activation
        x = F.relu(self.fc1(x))  # [256, 128 * 5 * 5] -> [256, 128]
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
    

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Error"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd, nkhd--> nhqk", [queries, keys])
        # energy shape -> query len is target source sentence, key len is source sentence
        # for each word in the target sentence, we have a score for each word in the source sentence

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        # attenthion = softmax(QK^T / sqrt(d_k))
        # dim = normalize across key length

        out = torch.einsum("nhql, nlhd--> nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape -> (N, heads, query_len, key_len)
        # value shape -> (N, value_len, heads, head_dim)

        out = self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion*embed_size), nn.ReLU(), nn.Linear(forward_expansion*embed_size, embed_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention+query))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.poition_embedding = nn.Embedding(max_length, embed_size)   

        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)])

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length =  x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cuda", max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1). unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
         


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
        if self.predict != "class":
            x, y, c = batch
        else:
            x, y = batch

        z = self.forward(x)
        loss = self.loss(z, y)
        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        # validation
        if self.predict != "class":
            x, y, c = batch
        else:
            x, y = batch

        z = self.forward(x)
        loss = self.loss(z, y) 
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    
    def test_step(self, batch, batch_idx, loss_name ='test_loss'):

        #test
        if self.predict != "class":
            x, y, c = batch
        else:
            x, y = batch

        z = self.forward(x)
        loss = self.loss(z, y) 

        if self.predict == "class":
        # std. label
            _, predicted = torch.max(z.data, 1) 
            test_acc = torch.sum(y == predicted).item() / (len(y)*1.0) 

        elif self.predict == "dowker":
            # invariant
            true = 0
            false = 0
            el = (y-z)

            for idx, i in enumerate(el):
                x = torch.round(i)
                if torch.sum(x) == 0.0:
                    true += 1
                else:
                    false += 1

            test_acc = true / (true + false)

        else:
        # invariant
            true = 0
            false = 0
            total_errors  = []
            el = ((y-z)/y)

            true50 = 0
            false50 = 0

            for idx, i in enumerate(el):
                total_errors.append(i)
                if i < 0.05:
                    true += 1
                else:
                    false += 1

            test_acc = true / (true + false)

        with open(f'5_class_errors_{self.predict}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in i:
                writer.writerow([item])

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
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
        optimizer = optim.Adam(model.parameters(), lr=0.000001)
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
        optimizer = optim.Adam(model.parameters(), lr=0.000001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice; 'adam' or 'sgd'.")

    return model, loss_fn, optimizer


