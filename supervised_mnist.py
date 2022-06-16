
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

from utils import *
from twolayernetwork import *


seed = 0
n_neurons = 100
n_train = 500
n_test = 100
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


width = 128 #34
kernel = 4
width = int(width/kernel)

# Sets up Gpu use
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

n_classes = 11
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / n_classes)

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015(
    n_inpt=width*width*2,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-10, 1e-3],  # 0.711
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(2, width, width),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time, device=device)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time, device=device)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Load MNIST data.
trainloader, testloader = createDataloaders(1)

# Create a dataloader to iterate and batch data
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons, device=device)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Labels to determine neuron assignments and spike proportions and estimate accuracy
labels = torch.empty(update_interval, device=device)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

# Train the network.
print("Begin training.\n")

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes = None
voltage_ims = None

pbar = tqdm(total=n_train)
pool = torch.nn.MaxPool2d(kernel)
for epoch in range(10):
    for i, (image, label) in enumerate(trainloader):
        if i > n_train:
            break

        image = pool(image.squeeze())
        image = torch.unsqueeze(image, 1)

        if i % update_interval == 0 and i > 0:
            # Get network predictions.
            all_activity_pred = all_activity(spike_record, assignments, n_classes)
            proportion_pred = proportion_weighting(
                spike_record, assignments, proportions, n_classes
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval
            )
            accuracy["proportion"].append(
                100 * torch.sum(labels.long() == proportion_pred).item() / update_interval
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]))
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spike_record, labels, n_classes, rates
            )

        # Add the current label to the list of labels for this update_interval
        labels[i % update_interval] = label[0]

        # Run the network on the input.
        # TODO is the clamp the supervisor??
        choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
        clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
        if gpu:
            inputs = {"X": image.cuda().view(time, 1, 1, width, width)}
        else:
            inputs = {"X" :image.reshape([image.size(0), 1, 2, width, width])}
        network.run(inputs=inputs, time=time, clamp=clamp)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

        network.reset_state_variables()  # Reset state variables.
        pbar.set_description_str("Train progress: ")
        pbar.update()

print("Progress: %d / %d \n" % (n_train, n_train))
print("Training complete.\n")

print("Testing....\n")

# Load MNIST data.


# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros(1, int(time / dt), n_neurons, device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)

pbar = tqdm(total=n_test)
for step, batch in enumerate(testloader):
    if step > n_test:
        break
    # Get next input sample.
    image = pool(batch[0].squeeze())
    image = torch.unsqueeze(image, 1)
    inputs = {"X": image.view(image.size(0), 1, 2, width, width)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch[1], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.

    pbar.set_description_str(
        f"Accuracy: {(max(accuracy['all'] ,accuracy['proportion'] ) / (step+1)):.3}"
    )
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))


print("Testing complete.\n")
