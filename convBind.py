### Toy example to test Conv3dConnection (the dataset used is MNIST but with a dimension replicated
### for each image (each sample has size (28, 28, 28))

import argparse
import os
from time import time as t

import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Connection, Conv2dConnection

from utils import *


seed = 0
n_neurons = 2700*2
n_train = 500
n_epochs = 10
n_test = 150
n_clamp = 1
exc = 22.5
inh = 120
theta_plus = 0.05
time = 200
dt = 1.0
intensity = 32
progress_interval = 10
update_interval = 25
train = True
plot = False
gpu = False
device_id = 0


batch_size = 1
kernel_size = 16
stride = 4
n_filters = 24
padding = 0

width = 128 #34
kernel = 4
#width = int(width/kernel)

trainloader, testloader = createDataloaders(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if not train:
    update_interval = n_test

conv_size = int((32 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 11)

# Build network.
network = Network()
input_layer = Input(n=2048, shape=(2, 32, 32), traces=True)

conv_layer = DiehlAndCookNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=PostPre,
    norm=0.4 * kernel_size**2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
)

w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[fltr1, i, j, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network.to("cuda")



spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None

pool = torch.nn.MaxPool2d(kernel)
# Train the network.
print("\nBegin training.\n")


for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    for step, (image, label) in enumerate(trainloader):
        if step > n_train:
            break
        # Get next input sample.
        image = pool(image.squeeze())
        image = torch.unsqueeze(image, 1)
        inputs = {
            "X": image.view(image.size(0), 1, 2, 32,32)}

        # Get next input sample (expanded to have shape (time, batch_size, 1, 28, 28))
        # if step > n_train:
        #     break
        # inputs = {
        #     "X": batch["encoded_image"]
        #     .view(time, batch_size, 1, 28, 28)
        #     .unsqueeze(3)
        #     .repeat(1, 1, 1, 28, 1, 1)
        #     .float()
        # }
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        #label = batch["label"]

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")