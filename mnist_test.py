import time

import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
import torchvision
from snntorch import functional as SF
from snntorch import surrogate
from snntorch import utils
from torch.utils.data import DataLoader

import snntorch.spikeplot as splt
from IPython.display import HTML

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
                                      torchvision.transforms.RandomRotation([-10, 10])])

# cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

# no augmentations for the testset
# cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

"""
#####################################################
TRAINING OUR NETWORK USING FRAMES CREATED FROM EVENTS
#####################################################
"""
# transform = tonic.transforms.Compose([torch.from_numpy,torchvision.transforms.RandomRotation([-10,10])])

# cached_trainset = CachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

# no augmentations for the testset
# cached_testset = CachedDataset(testset, cache_path='./cache/nmnist/test')

batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=False)


n = 125

# index into a single sample and sum the on/off channels
a = (trainloader.dataset[n][0][:, 0] + trainloader.dataset[n][0][:, 1])

#  Plot
fig, ax = plt.subplots()
anim = splt.animator(a, fig, ax, interval=10)
HTML(anim.to_html5_video())

anim.save('dvsgesture_animator.mp4', writer = 'ffmpeg', fps=50)

"""
DEFINING OUR NETWORK
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# network parameters
num_inputs = 28 * 28
num_hidden = 128
num_outputs = 10
num_steps = 1

# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid()

#  Initialize Network
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(34 * 34 * 2, 10),
                    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
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
num_iters = 500

loss_hist = []
acc_hist = []
t0 = time.time()
# training loop
for epoch in range(num_epochs):

    for i, (data, targets) in enumerate(iter(trainloader)):
        data = data.to(device)

        targets = targets.to(device)
        print(data.size(), targets.size())
        # print(net)
        net.train()
        # sprint(net)
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

test_loss_hist = []
test_acc_hist = []
net.eval()
for i, (data, targets) in enumerate(iter(testloader)):
    data = data.to(device)
    targets = targets.to(device)
    print(data.size(), targets.size())
    spk_rec = forward_pass(net, data)
    loss_val = loss_fn(spk_rec, targets)

    # Gradient calculation + weight update
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    # Store loss history for future plotting
    test_loss_hist.append(loss_val.item())

    print(f"Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

    acc = SF.accuracy_rate(spk_rec, targets)
    test_acc_hist.append(acc)
    print(f"Accuracy: {acc * 100:.2f}%\n")

    # training loop breaks after 50 iterations
    if i == num_iters:
        break

print(np.mean(test_acc_hist))
# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
