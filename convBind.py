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
from bindsnet.evaluation import *
import numpy as np


from utils import *

seed = 0
n_neurons = 2700 * 2
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

width = 34  # 34
kernel = 1
# width = int(width/kernel)

trainloader, testloader = createMNISTDataloaders(1)

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

conv_size = int((34 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network = Network()
input_layer = Input(n=2312, shape=(2, 34, 34), traces=True)

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
    norm=0.4 * kernel_size ** 2,
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

# Record spikes during the simulation.
spike_record = torch.ones((update_interval, time, n_filters, conv_size, conv_size), device=device)

# Neuron assignments and spike proportions.
n_classes = 11
assignments = -torch.ones(n_filters, conv_size, conv_size, device=device)
proportions = torch.zeros((n_filters, conv_size, conv_size, n_classes), device=device)
rates = torch.zeros((n_filters, conv_size, conv_size, n_classes), device=device)

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
# Sequence of accuracy estimates.
accuracy = {"top1": [], "top3": []}


for epoch in range(n_epochs):
    labels = []
    pbar = tqdm(total=n_train)
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    for step, (image, label) in enumerate(trainloader):
        if step > n_train:
            break
        # Get next input sample.
        # image = pool(image.squeeze())
        # image = torch.unsqueeze(image, 1)
        # image = image[:666]
        # print(image.shape)
        # image = torch.unsqueeze(image, 1)
        inputs = {"X": image.view(image.size(0), 1, 2, 34, 34)}

        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        # label = batch["label"]

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(label, device=device)

            # Get network predictions.

            acc_spikes = spike_record.sum(1)
            acc_rates = torch.zeros((spike_record.size(0), n_classes))
            for i in range(n_classes):
                # Count the number of neurons with this label assignment.
                n_assigns = torch.sum(assignments == i).float()

                if n_assigns > 0:
                    # Get indices of samples with this label.
                    indices_acc = torch.nonzero(assignments == i).view(-1)
                    # Compute layer-wise firing rate for this label.
                    acc_rates[:, i] = torch.sum(acc_spikes[:, indices_acc], 1) / n_assigns
            top3preds = torch.sort(acc_rates, dim=1, descending=True)[1][:, :3]
            top3acc = 0
            top1acc = 0
            for i in range(len(label_tensor)):
                # print(label_tensor[i], top3preds[i])
                if label_tensor[i]==top3preds[i][0]:
                    top1acc +=1
                    top3acc += 1
                elif label_tensor[i] in top3preds[i]:
                    top3acc += 1

            top3acc /= len(label_tensor)
            top1acc /= len(label_tensor)
            #all_activity_pred = all_activity(
             #   spikes=spike_record, assignments=assignments, n_labels=n_classes
            #)

            # Compute network accuracy according to available classification strategies.
            accuracy["top1"].append(
                100 * top1acc
            )

            accuracy["top3"].append(
                100 * top3acc
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["top1"][-1],
                    np.mean(accuracy["top1"]),
                    np.max(accuracy["top1"]),
                )
            )
            print(
                "Top 3 accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["top3"][-1],
                    np.mean(accuracy["top3"]),
                    np.max(accuracy["top3"]),
                )
            )


            labels = []

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)
        stepudpate=step % update_interval
        spikeyy = spikes["Y"].get("s")
        spike_record[stepudpate] = spikeyy.squeeze()

        network.reset_state_variables()  # Reset state variables.
        labels.append(label)
        pbar.set_description_str("Train progress: ")
        pbar.update()

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")
