import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NNModel(nn.Module):
    def __init__(self, input_shape, output_shape, norm):
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

def setup_NN(input_shape, output_shape, opt, norm):
    model = NNModel(input_shape, output_shape, norm)

    # loss function compares y_pred to y_true: in our case cross-entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # optimizer; i think we only use adam
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice. Choose 'adam' or 'sgd'.")

    return model, loss_fn, optimizer

