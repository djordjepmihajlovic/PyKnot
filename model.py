import torch
import torch.nn as nn
import torch.optim as optim

class NNModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_activation, norm):
        super(NNModel, self).__init__()

        self.flatten_layer = nn.Flatten()

        if norm:
            self.bn_layer = nn.BatchNorm1d(input_shape[0])
            self.dense_layer1 = nn.Linear(input_shape[0]*input_shape[1], 320)
            self.activation1 = nn.ReLU()
        else:
            self.dense_layer1 = nn.Linear(input_shape[0]*input_shape[1], 320)
            self.activation1 = nn.ReLU()

        # Add hidden layers to NN
        self.dense_layer2 = nn.Linear(320, 320)
        self.activation2 = nn.ReLU()

        self.dense_layer3 = nn.Linear(320, 320)
        self.activation3 = nn.ReLU()

        self.dense_layer4 = nn.Linear(320, 320)
        self.activation4 = nn.ReLU()

        # Add output layer
        self.output_layer = nn.Linear(320, output_shape)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten_layer(x)

        if hasattr(self, 'bn_layer'):
            x = self.bn_layer(x)

        x = self.dense_layer1(x)
        x = self.activation1(x)

        x = self.dense_layer2(x)
        x = self.activation2(x)

        x = self.dense_layer3(x)
        x = self.activation3(x)

        x = self.dense_layer4(x)
        x = self.activation4(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x

def setup_NN(input_shape, output_shape, hidden_activation, opt, norm):
    model = NNModel(input_shape, output_shape, hidden_activation, norm)

    # loss function compares y_pred to y_true: in this case sparse categorical cross-entropy
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer choice. Choose 'adam' or 'sgd'.")

    return model, loss_fn, optimizer


