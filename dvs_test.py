from net import *
from utils import *

"""
Feedforward NN with LIF
based on Tutorial 7 SNNTorch
"""

batch_size = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# network parameters
# for NMNIST
# trainloader, testloader = createMNISTDataloaders(batch_size)
# num_inputs = 2 * 34 * 34  # width*height*channels (on-spikes for luminance increasing; off-spikes for luminance decreasing)
# num_hidden = 2 * 34 * 34
# num_classes = 10
# kernel = 1

# for DVS Gesture
# Number of train samples: 1176
# Number of test samples: 288
trainloader, testloader = createDataloaders(batch_size)
kernel = 4
num_inputs = int(128 * 128 * 2 / (kernel*kernel)) # height*channels*width (on-spikes for luminance increasing; off-spikes for luminance decreasing)
num_hidden = 128 * 128 * 2
num_classes = 11




# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid(slope=25)

# Epochs & Iterations
num_epochs = 10
num_train = 50
num_test = 15  # 15 => maximum amount of steps in DVS Gesture if batchsize 16


# net = FFNet(num_inputs, num_classes, beta).to(device)
net = RFFNet(num_inputs, num_classes, beta).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
# Cross Entropy encourages the correct class to fire at all time steps, and aims to suppress incorrect classes from firing
#loss_fn = SF.ce_rate_loss() # Cross Entropy Spike Rate Loss, applies the Cross Entropy function at every time step
#loss_fn = SF.ce_count_loss() # Cross Entropy Spike Count Loss, accumulates spikes first & applies Cross Entropy Loss only once
# loss_fn = SF.ce_max_membrane_loss() # encourages the maximum membrane potential of the correct class to increase, while suppressing the maximum membrane potential of incorrect classes

loss_fn = SF.mse_count_loss(correct_rate=1, incorrect_rate=0.8) #default rate correct=1, incorrect=1, tutorial: 0.8,0.2 -> avoid dead neurones
#loss_fn = SF.mse_membrane_loss()


train_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss"])
test_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss"])

# training loop
t0 = time.time()
flatten = nn.Flatten(2, 4)
pool = nn.MaxPool3d(kernel_size=(1,int(kernel),int(kernel)))

for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(trainloader, 0):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec = net(flatten(pool(data)))
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        hist = evaluate(spk_rec, targets, t0, loss_val, epoch, i, train=True)
        train_hist = pd.concat([train_hist, hist])
        t0 = time.time()

        # training loop breaks after x iterations
        if i == num_train:
            break

    for i, (data, targets) in enumerate(testloader, 0):
        # data = data.to(device)
        targets = targets.to(device)
        net.eval()

        # print(step_data.shape, compact.shape, step_data.view(batch_size, -1).shape)
        spk_rec = net(flatten(pool(data)))

        loss_val = loss_fn(spk_rec, targets)

        # Store loss history for future plotting
        hist = evaluate(spk_rec, targets, t0, loss_val, epoch, i, train=False)
        test_hist = pd.concat([test_hist, hist])
        t0 = time.time()
        if i == num_test:
            break

    #print parameter weights after each epoch.
    for param in net.parameters():
        print(param.data)

train_hist.to_csv('train.csv')
test_hist.to_csv('test.csv')
