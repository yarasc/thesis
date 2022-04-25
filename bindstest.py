import os
from time import time as t

import numpy as np
import tonic
import tonic.transforms as transforms
import torch
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.network.monitors import Monitor
from torch.utils.data import DataLoader
from tqdm import tqdm

from twolayernetwork import TwoLayerNetworkShaped
from utils import createDataloaders

batch_size = 16


# network parameters
n_classes = 11
num_inputs = 128 * 128 * 2  # width*height*channels (on-spikes for luminance increasing; off-spikes for luminance decreasing)
num_hidden = 128 * 128 * 2
num_steps = 1

# Epochs & Iterations
num_epochs = 10
num_test = 50
num_train = 500

seed = 0

# BINDSNET specific parameters
n_workers = -1
update_steps = 256
theta_plus = 0.05
time = 10
dt = 1.0
intensity = 128


progress_interval = 4
update_interval = update_steps * batch_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gpu = False
torch.manual_seed(seed)
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()
n_sqrt = int(np.ceil(np.sqrt(num_inputs)))
start_intensity = intensity

network = TwoLayerNetworkShaped(
    n_inpt=num_inputs,
    n_neurons=num_inputs,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    inpt_shape=(2, 128, 128)
)

trainloader, testloader = createDataloaders(batch_size)
# Neuron assignments and spike proportions.

assignments = -torch.ones(num_inputs, device=device)
proportions = torch.zeros((num_inputs, n_classes), device=device)
rates = torch.zeros((num_inputs, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

spike_record = torch.zeros((500, int(time / dt), num_inputs), device=device)

# Train the network.
print("\nBegin training.\n")
start = t()

for epoch in range(num_epochs):
    labels = []
    # Create a dataloader to iterate and batch dvs_data
    pbar_training = tqdm(total=num_train)
    for step, (data, targets) in enumerate(trainloader):
        print("step", step)
        if step > num_train:
            break

        print("run network")
        # Run the network on the input.
        # unsupervised
        network.run(inputs={"X": data}, time=time, input_time_dim=1)

        print("predict activity")
        # Get network predictions.
        targets.to(device)
        all_activity_pred = all_activity(spikes=spike_record, assignments=assignments, n_labels=n_classes)
        #proportion_pred = proportion_weighting(spikes=spike_record,assignments=assignments,proportions=proportions,n_labels=n_classes,)

        # Compute network accuracy according to available classification strategies.
        acc = 0
        for label in range(len(targets)):
            if targets[label]==all_activity_pred[label]:
                acc+=1
        acc = 100*acc/len(targets)
        print(acc)
        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(spikes=spike_record,labels=targets,n_labels=n_classes,rates=rates,)

        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)

print("Training complete.\n")

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=num_test)
for step, (data, targets) in enumerate(testloader):
    if step > num_test:
        break
    # Get next input sample.
    for frame in data:
        # Run the network on the input.
        network.run(inputs={"X": frame}, time=time, input_time_dim=1)

        # Add to spikes recording.
        spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

        # Convert the array of labels into a tensor

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
        accuracy["all"] += float(torch.sum(targets.long() == all_activity_pred).item())
        accuracy["proportion"] += float(torch.sum(targets.long() == proportion_pred).item())

        network.reset_state_variables()  # Reset state variables.
        pbar.set_description_str("Test progress: ")
        pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / num_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / num_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, num_epochs, t() - start))
print("Testing complete.\n")
