from net import *
from utils import *
from snntorch import surrogate


"""
Feedforward NN with LIF
based on Tutorial 7 SNNTorch
"""

batch_size = 8

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Pick Dataset and corresponding parameters
"""
# network parameters
# for NMNIST
# trainloader, testloader = createMNISTDataloaders(batch_size)
# num_inputs = 2 * 34 * 34  # width*height*channels
# num_hidden = 2 * 34 * 34
# num_classes = 10
# kernel = 1
# time = 200

# for DVS Gesture
# Number of train samples: 1176
# Number of test samples: 288
trainloader, testloader = createDataloaders(batch_size)
kernel = 4
num_inputs = int(128 * 128 * 2 / (kernel * kernel))  # height*channels*width
num_hidden = int(128 * 128 * 2 / (kernel * kernel))
num_classes = 11
timesteps = int(2000 * 1000 / 3000) # learning time in ms * conversion to micros / toFrame time-window


# TRY alternative filter
# trainloader, testloader = dataloaderWithSpecTransform(batch_size, transform="averagedtimesurface")
# kernel = 4
# num_inputs = int(128 * 128 * 2 / (kernel * kernel))  # height*channels*width
# num_hidden = int(128 * 128 * 2 / (kernel * kernel))
# num_classes = 11

# printEventData(1)

# spiking neuron parameters
beta = 0.4  # neuron decay rate (0.9 in tutorial 5, 0.5 in other tutorials)
grad = surrogate.fast_sigmoid(slope=25) #default = 25, the higher the steeper the curve

# Epochs & Iterations
num_epochs = 10
num_train = 133
num_test = 30  # 15 => maximum amount of steps in DVS Gesture if batchsize 16

"""
Choose model (simple forward, or with RLeaky)
"""
# net = FFNet(num_inputs, num_classes, beta, grad).to(device)
# net = TwoFFNet(num_inputs, num_hidden, num_classes, beta, grad).to(device)
# net = RFFNet(num_inputs, num_classes, beta, grad).to(device)
# net = TwoRFFNet(num_inputs, num_classes, beta, grad).to(device)
net = CNNet(num_classes, beta, grad, batch_size, filter=32, kernel=2).to(device)

summary(net, (2, 32, 32))
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))

# Cross Entropy encourages the correct class to fire at all time steps,
# and aims to suppress incorrect classes from firing
# loss_fn = SF.ce_rate_loss() # Cross Entropy Spike Rate Loss, applies the Cross Entropy function at every time step
# loss_fn = ce_rate_loss_8() # Cross Entropy Spike Rate Loss, applies the Cross Entropy function at every time step
# loss_fn = SF.ce_count_loss() # Cross Entropy Spike Count Loss, accumulates spikes first & applies CE only once

# loss_fn = SF.mse_membrane_loss()
#loss_fn = SF.mse_temporal_loss(multi_spike=False, off_target=)
loss_fn = SF.mse_count_loss(correct_rate=1, incorrect_rate=0.8
                            )
# default rate correct=1, incorrect=1, tutorial: 0.8,0.2 -> avoid dead neurones


train_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss", "Top3"])
test_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss",  "Top3"])

# training loop
t0 = time.time()
flatten = nn.Flatten(2, 4)
pool = nn.MaxPool3d(kernel_size=(1, int(kernel), int(kernel)))

for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(trainloader, 0):
        data = data.to(device)
        targets = targets.to(device)

        """
        0 1: hand clap
        1 2: right hand wave
        2 3: left hand wave
        3 4: right arm clockwise
        4 5: right arm counter clockwise
        5 6: left arm clockwise
        6 7: left arm counter clockwise
        7 8: arm roll
        8 9: air drums
        9 10: air guitar
        10 11: other gestures
        """

        net.train()
        # [timesteps, batchsize, channel, width, width]
        data = pool(data)
        # print(data.shape)
        data = data[:timesteps, ]
        # [timesteps, batchsize, channel, width/kernel, width/kernel]
        # data = flatten(data) #TODO if cnn take out flatten
        # [timesteps, batchsize, channel * width * width]
        spk_rec = net(data)
        #print(data.shape)
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

        data = pool(data)
        # [timesteps, batchsize, channel, width/kernel, width/kernel]
        data = data[:timesteps, ]
        # data = flatten(data) #TODO if cnn take out flatten
        # [timesteps, batchsize, channel * width * width]

        spk_rec = net(data)
        loss_val = loss_fn(spk_rec, targets)
        # Store loss history for future plotting
        hist = evaluate(spk_rec, targets, t0, loss_val, epoch, i, train=False)
        test_hist = pd.concat([test_hist, hist])


        t0 = time.time()
        if i == num_test:
            break

    # print parameter weights after each epoch.
    # for param in net.parameters():
    #     print(param.data)

train_hist.to_csv('traindvscnntmsetemp.csv')
test_hist.to_csv('testdvscnntmsetemp.csv')
