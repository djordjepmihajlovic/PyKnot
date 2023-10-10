import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# nn.Linear applies linear transformation y = x*A^T + b

class NNmodel(nn.Module):

    #  Input layer (features) -->
    #  Hidden Layer1 (x no. of neurons) --> ... --> 
    #  Output (class of knot)

    def __init__(self, in_features, h1, out_features, norm):
        super().__init__() # instantiate nn.Module

        # 4 hidden layers, fc1 --> fc4
        if norm:
            in_features_norm = nn.BatchNorm2d(in_features)

            self.fc1 = nn.Linear(in_features_norm, h1)

        else:
            self.fc1 = nn.Linear(in_features, h1)

        self.fc2 = nn.Linear(h1, h1)
        self.fc3 = nn.Linear(h1, h1)
        self.fc4 = nn.Linear(h1, h1)
        self.out = nn.Linear(h1, out_features)


    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.cross_entropy(self.out, dim = 1) # <-- not actually sure

        return x
    
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


#def build_model(hp, input_shape, output_shape, hidden_activation, norm):
    
    