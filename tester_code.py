import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True) 

# batching passes data 10 datapoints at a time (helps with load)
# shuffle shuffles; as in data is no longer ordered 1(),2(),3(),...,10()

# plt.imshow(data[0][0].view(28, 28))
# plt.show()

# balance of data -> should check to avoid local minima's in loss function

class Net(nn.Module):
    def __init__(self, input_size = 784, train_size = 64, output_size = 10):
        super().__init__() #basically runs nn.Module
        self.fc1 = nn.Linear(input_size, train_size)
        self.fc2 = nn.Linear(train_size, train_size)
        self.fc3 = nn.Linear(train_size, train_size)
        self.fc4 = nn.Linear(train_size, output_size)

    def forward(self, x): # define order + activation function (F.relu() in this case) + optimizer(multiclass=softmax(usually))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.cross_entropy(x, dim=1)


net = Net()

X = torch.rand((28, 28))

output = net(X.view(-1, 28*28)) #-1?

# loss and optimization

optimizer = optim.Adam(net.parameters(), lr = 0.001) #learning rate = size of steps to take (try find 'lowest minima')

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of features and labels
        X, y = data
        net.zero_grad()  # note: pytorch!
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y) # note: one-hot vector [0, 0, 1, 0] use mean sq
        loss.backward() # backpropagation ***
        optimizer.step() # adjust weights

    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total +=1
print("Accuracy: ", round(correct/total, 3))

plt.imshow(X[1].view(28,28))
plt.show()

print(torch.argmax(net(X[1].view(-1,784))[0]))













