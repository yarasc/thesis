from snntorch import surrogate

from net import *
from utils import *
from torchsummary import summary

"""
Feedforward NN with LIF
based on Tutorial 7 SNNTorch
"""


def run_snntorch(name,
                 num_train,
                 num_test,
                 batch_size=8,
                 kernel=4,
                 section=2000,
                 beta=0.4,
                 model="CNN",
                 loss="mse",
                 set="dvs",
                 cor_rate=1,
                 incor_rate=0.8
                 ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    """
    Pick Dataset and corresponding parameters
    """
    # network parameters
    # for NMNIST
    if set == "dvs":
        # Number of train samples: 1176
        # Number of test samples: 288
        print("DVS Gesture; batch size", batch_size, "; kernel", kernel, "; time section", section)
        trainloader, testloader = createDataloaders(batch_size)
        num_inputs = int(128 * 128 * 2 / (kernel * kernel))  # height*channels*width
        num_hidden = int(128 * 128 * 2 / (kernel * kernel))
        num_classes = 10
        timesteps = int(section * 1000 / 3000)  # learning time in ms * conversion to micros / toFrame time-window
    else:
        print("MNIST; batch size", batch_size)
        trainloader, testloader = createMNISTDataloaders(batch_size)
        num_inputs = 2 * 34 * 34  # width*height*channels
        num_hidden = 2 * 34 * 34
        num_classes = 10
        kernel = 1
        timesteps = 0

    # spiking neuron parameters
    # neuron decay rate (0.9 in tutorial 5, 0.5 in other tutorials)
    grad = surrogate.fast_sigmoid(slope=25)  # default = 25, the higher the steeper the curve

    # Epochs & Iterations
    num_epochs = 10
    """
    Choose model (simple forward, or with RLeaky)
    """

    if model == "SimpRNN":
        print("no Hidden Layer RNN")
        net = RFFNet(num_inputs, num_classes, beta, grad).to(device)
    elif model == "CNN":
        print("CNN")
        if set=="dvs":
            net = CNNet(num_classes, beta, grad, batch_size, filter=32, kernel=2,lin_size=225).to(device)
        else:
            net= CNNet(num_classes, beta, grad, batch_size, filter=32, kernel=2,lin_size=256)
    else:
        print("1 Hidden Layer RNN")
        net = TwoRFFNet(num_inputs, num_hidden, num_classes, beta, grad).to(device)
    # summary(net, (8, 2, 32, 32))
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))

    if loss == "mse":
        print("Loss MSE", cor_rate, incor_rate)
        loss_fn = SF.mse_count_loss(correct_rate=cor_rate, incorrect_rate=incor_rate)
    else:
        print("Loss CE")
        loss_fn = SF.ce_rate_loss()
        # default rate correct=1, incorrect=1, tutorial: 0.8,0.2 -> avoid dead neurones

    train_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss", "Top3"])
    test_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Loss", "Top3"])

    # training loop
    t0 = time.time()
    flatten = nn.Flatten(2, 4)
    pool = nn.MaxPool3d(kernel_size=(1, int(kernel), int(kernel)))

    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(trainloader, 0):
            data = data.to(device)  # [timesteps, batchsize, channel, width, width]
            targets = targets.to(device)
            net.train()

            # PREPARE Data
            data = pool(data)
            if set == "dvs":
                data = data[:timesteps, ]  # [timesteps, batchsize, channel, width/kernel, width/kernel]
            if model != "CNN":
                data = flatten(data)  # [timesteps, batchsize, channel * width * width]
            spk_rec = net(data)
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
            data = data.to(device)  # [timesteps, batchsize, channel, width, width]
            targets = targets.to(device)
            net.eval()

            # PREPARE Data
            data = pool(data)
            if set == "dvs":
                data = data[:timesteps, ]  # [timesteps, batchsize, channel, width/kernel, width/kernel]
            if model != "CNN":
                data = flatten(data)  # [timesteps, batchsize, channel * width * width]

            spk_rec = net(data)
            loss_val = loss_fn(spk_rec, targets)

            # Store loss history for future plotting
            hist = evaluate(spk_rec, targets, t0, loss_val, epoch, i, train=False)
            test_hist = pd.concat([test_hist, hist])

            t0 = time.time()
            if i == num_test:
                break

    train_hist.to_csv(str('train' + name + '.csv'))
    test_hist.to_csv(str('test' + name + '.csv'))
