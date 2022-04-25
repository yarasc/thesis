import time

import pandas as pd
import tonic
import tonic.transforms as transforms
import torchvision
from snntorch import functional as SF
from snntorch import surrogate
from torch.utils.data import DataLoader

from net import *

"""
Feedforward NN with LIF
based on Tutorial 7 SNNTorch
"""

sensor_size = tonic.datasets.DVSGesture.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=3000)
                                      ])

trainset = tonic.datasets.DVSGesture(save_to='./dvs_data', transform=frame_transform, train=True)
testset = tonic.datasets.DVSGesture(save_to='./dvs_data', transform=frame_transform, train=False)

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10, 10])])

batch_size = 16
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# network parameters
num_inputs = 128 * 128 * 2  # width*height*channels (on-spikes for luminance increasing; off-spikes for luminance decreasing)
num_hidden = 128 * 128 * 2
num_outputs = 11
num_steps = 1

# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid()

net = FFNet(num_inputs, num_outputs, beta).to(device)

"""
TRAINING
"""
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
#loss_fn = SF.ce_rate_loss()
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 10
num_iters = 50

train_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss"])
test_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss"])

# training loop
t0 = time.time()
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(trainloader, 0):
        step = 0
        loss_hist = []
        acc_hist = []

        data = data.to(device)
        targets = targets.to(device)

        net.train()
        # compute prediction and loss
        # spk_rec = forward_pass(net, frame)
        spk_rec = []
        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out = net(data[step].view(batch_size, -1))
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec)
        spk_rec = spk_rec[:,:,-1,:]
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        acc = SF.accuracy_rate(spk_rec, targets)
        # Store loss history for future plotting every 0.3 sec
        x = [[epoch, i, acc, loss_val.item()]]
        tmp_df = pd.DataFrame(x, columns=["Epoch", "Iteration", "Accuracy", "Loss"])
        train_hist = pd.concat([train_hist, tmp_df])
        print('{} s'.format(time.time() - t0), end=": ")
        t0 = time.time()
        print(f"Epoch {epoch}, Iteration {i} – Train Loss: {loss_val.item():.2f}", end=" – ")
        print(f"Accuracy: {acc * 100:.2f}%")

        # training loop breaks after 50 iterations
        if i == 2:
            break

    for i, (data, targets) in enumerate(testloader, 0):
        test_loss_hist = []
        test_acc_hist = []
        # frame = frame.reshape([1, batch_size, 2, 128, 128])
        data = data.to(device)
        targets = targets.to(device)

        net.eval()
        # spk_rec = forward_pass(net, frame)
        #spk_rec, mem_rec = net(frame)
        spk_rec = []
        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out = net(data[step].view(batch_size, -1))
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec, dim=0)
        spk_rec = spk_rec[:,:,-1,:]
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting every 0.3 second
        acc = SF.accuracy_rate(spk_rec, targets)
        x = [[epoch, i, acc, loss_val.item()]]
        tmp_df = pd.DataFrame(x, columns=["Epoch", "Iteration", "Accuracy", "Loss"])
        test_hist = pd.concat([test_hist, tmp_df])

        print('{} s'.format(time.time() - t0), end=": ")
        t0 = time.time()
        print(f"Epoch {epoch}, Iteration {i} – Test Loss: {loss_val.item():.2f}", end=" – ")
        print(f"Accuracy: {acc * 100:.2f}%")

        # training loop breaks after 50 iterations
        if i == 20:
            break

train_hist.to_csv('train.csv')
train_hist.to_csv('test.csv')

# x = np.arange(num_epochs)
# fig = plt.figure(facecolor="w")
# plt.errorbar(y=test_hist.accuracy, x=x, yerr=test_hist.error_acc, label='Test/Val')
# plt.errorbar(y=train_hist.accuracy, x=x, yerr=train_hist.error_acc, label='Train')
# plt.title("Average Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(loc='lower right')
# plt.show()
# # plt.savefig("accuracy.png", dpi=150)
#
#
# fig = plt.figure(facecolor="w")
# plt.errorbar(y=test_hist.loss, x=x, yerr=test_hist.error_los, label='Test/Val')
# plt.errorbar(y=train_hist.loss, x=x, yerr=train_hist.error_loss, label='Train')
# plt.title("Average Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend(loc='lower right')
# plt.show()
# # plt.savefig("loss.png", dpi=150)
