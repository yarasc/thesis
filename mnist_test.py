import matplotlib.pyplot as plt
import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset

import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate
from snntorch import utils

from pytorch_model_summary import summary

"""
#########################################
USING TONIC TO LOAD NEUROMORPHIC DATASETS
#########################################
"""
dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
events, target = dataset[0]

"""
TRANSFORMATIONS
"""
sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events

# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=1000)
                                      ])

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

# no augmentations for the testset
cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')


"""
#####################################################
TRAINING OUR NETWORK USING FRAMES CREATED FROM EVENTS
#####################################################
"""
# transform = tonic.transforms.Compose([torch.from_numpy,torchvision.transforms.RandomRotation([-10,10])])

# cached_trainset = CachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

# no augmentations for the testset
# cached_testset = CachedDataset(testset, cache_path='./cache/nmnist/test')

batch_size = 128
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors)

"""
DEFINING OUR NETWORK
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# network parameters
num_inputs = 28*28
num_hidden = 128
num_outputs = 10
num_steps = 1

# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid()

#  Initialize Network
net = nn.Sequential(nn.Linear(34, 12), #Conv2d(input,weight,stride) #Linear(input,output)
                    #nn.MaxPool2d(2), #kernel size (stride)
                    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                    nn.Linear(12, 32),
                    #nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(34*8*8, 10),
                    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True),

                    ).to(device)


# this time, we won't return membrane as we don't need it

def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


"""
TRAINING
"""
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.ce_rate_loss()

num_epochs = 1
num_iters = 50

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(trainloader)):
        data = data.to(device)

        targets = targets.to(device)
        print(data.size(), targets.size())
        #print(net)
        net.train()
        #sprint(net)
        spk_rec = forward_pass(net, data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%\n")

        # training loop breaks after 50 iterations
        if i == num_iters:
            break

"""
#######
RESULTS
#######
"""
# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
