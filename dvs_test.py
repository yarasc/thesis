from snntorch import surrogate

from net import *
from utils import *

"""
Feedforward NN with LIF
based on Tutorial 7 SNNTorch
"""

batch_size = 16

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# network parameters
num_inputs = 128 * 128 * 2  # width*height*channels (on-spikes for luminance increasing; off-spikes for luminance decreasing)
num_hidden = 128 * 128 * 2
num_classes = 11
num_steps = 1

# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid()

# Epochs & Iterations
num_epochs = 10
num_train = 50
num_test = 18

trainloader, testloader = createDataloaders(batch_size)
net = FFNet(num_inputs, num_classes, beta).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.ce_rate_loss()
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)


train_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss"])
test_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss"])

# training loop
t0 = time.time()

for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(trainloader, 0):
        data = data.to(device)
        targets = targets.to(device)

        net.train()

        spk_rec = []
        # compute prediction and loss
        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out = net(data[step].view(batch_size, -1))
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec)
        spk_rec = spk_rec[:, :, -1, :]
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        hist = evaluate(spk_rec, targets, t0, loss_val, epoch, i, train=True)
        train_hist = pd.concat([train_hist, hist])
        # training loop breaks after 50 iterations
        if i == 2: #num_train:
            break

    for i, (data, targets) in enumerate(testloader, 0):
        data = data.to(device)
        targets = targets.to(device)
        if i >= 15:
            print(data.shape)
        net.eval()

        spk_rec = []
        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out = net(data[step].view(batch_size, -1))
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec, dim=0)
        spk_rec = spk_rec[:, :, -1, :]
        loss_val = loss_fn(spk_rec, targets)

        # Store loss history for future plotting every 0.3 second
        hist = evaluate(spk_rec, targets, t0, loss_val, epoch, i, train=True)
        train_hist = pd.concat([train_hist, hist])

        if i == num_test:
            break

train_hist.to_csv('train.csv')
train_hist.to_csv('test.csv')
