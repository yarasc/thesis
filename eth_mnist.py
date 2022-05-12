import argparse
import os
from time import time as t

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
from bindsnet.evaluation import * #all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights


from utils import *

seed = 0
n_neurons = 100
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

width = 128 #34
kernel = 4
#width = int(width/kernel)

trainloader, testloader = createDataloaders(1)

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

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=int(width * width * 2 / (kernel*kernel)),
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(2, int(width/kernel), int(width/kernel)),
)

# Directs network to GPU
if gpu:
    network.to("cuda")



# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 11
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"top1": [], "top3": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

pool = torch.nn.MaxPool2d(kernel)

# Train the network.
print("\nBegin training.\n")

start = t()
for epoch in range(n_epochs):
    labels = []
    pbar = tqdm(total=n_train)
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data

    for step, (image, label) in enumerate(trainloader):
        if step > n_train:
            break
        # Get next input sample.
        image = pool(image.squeeze())
        image = torch.unsqueeze(image, 1)
        inputs = {"X": image.reshape([image.size(0), 1, 2, int(width/kernel), int(width/kernel)])}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
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
            top3preds = torch.sort(acc_rates, dim=1, descending=True)[1][:,:3]
            top3acc = 0
            for i in range(len(label_tensor)):
                #print(label_tensor[i], top3preds[i])
                if label_tensor[i] in top3preds[i]:
                    top3acc += 1
            top3acc/=len(label_tensor)

            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )
            # Compute network accuracy according to available classification strategies.
            accuracy["top1"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
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

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.append(label)

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()


        network.reset_state_variables()  # Reset state variables.
        pbar.set_description_str("Train progress: ")
        pbar.update()

print("Progress: %d / %d (%.4f seconds)" % (1 + 1, n_epochs, t() - start))
print("Training complete.\n")



# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(testloader):
    if step >= n_test:
        break
    # Get next input sample.
    image = pool(batch[0].squeeze())
    image = torch.unsqueeze(image, 1)
    inputs = {"X": image.view(image.size(0),1, 2, int(width/kernel), int(width/kernel))}
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
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))


print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")