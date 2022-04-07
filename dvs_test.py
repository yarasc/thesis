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
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

np.set_printoptions(threshold=100000)
import snntorch.spikeplot as splt
from IPython.display import HTML

"""
Feedforward NN with LIF
based on Tutorial 7 SNNTorch
"""

"""
#########################################
USING TONIC TO LOAD NEUROMORPHIC DATASETS
#########################################
"""
"""
TRANSFORMATIONS
"""
sensor_size = tonic.datasets.DVSGesture.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=3000)
                                      ])

trainset = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=False)

"""
#####################################################
TRAINING OUR NETWORK USING FRAMES CREATED FROM EVENTS
#####################################################
"""

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10, 10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/dvs/train')
cached_testset = DiskCachedDataset(testset, cache_path='./cache/dvs/test')

# no augmentations for the testset
# cached_testset = DiskCachedDataset(testset, cache_path='./cache/dvs/test')


batch_size = 16
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

n = 125
# index into a single sample and sum the on/off channels
# a = (trainloader.dataset[n][0][:, 0] + trainloader.dataset[n][0][:, 1])
# #  Plot
# fig, ax = plt.subplots()
# anim = splt.animator(a, fig, ax, interval=10)
# HTML(anim.to_html5_video())
# anim.save('dvsgesture_animator.mp4', writer = 'ffmpeg', fps=50)
"""
DEFINING OUR NETWORK
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# network parameters
num_inputs = 128 * 128 * 2  # width*height*channels (on-spikes for luminance increasing; off-spikes for luminance decreasing)
num_hidden = 128 * 128 * 2
num_outputs = 11
num_steps = 1

# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid()

#  Initialize Network
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_outputs),
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
num_iters = 100

loss_hist = []
acc_hist = []
test_loss_hist = []
test_acc_hist = []

# training loop
t0 = time.time()
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(trainloader, 0):
        data = data.to(device)
        targets = targets.to(device)
        # data = data.to(torch.bool)
        # targets = targets.to(torch.bool)
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
        print('{} seconds'.format(time.time() - t0))
        t0 = time.time()
        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%\n")

        # training loop breaks after 50 iterations
        if i == num_iters:
            break
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()

# with torch.no_grad():
net.eval()
for i, (data, targets) in enumerate(testloader, 0):
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

"""
#######
RESULTS
#######
"""
# Plot Loss
print(np.mean(test_acc_hist))

fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
