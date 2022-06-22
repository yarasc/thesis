import torch
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre, NoOp
from bindsnet.network.monitors import Monitor
from bindsnet.network import Network
from bindsnet.evaluation import *  #
import numpy as np
from utils import *

network = Network()

# Create two populations of neurons, one to act as the "source"
# population, and the other, the "target population".
# Neurons involved in certain learning rules must record synaptic
# traces, a vector of short-term memories of the last emitted spikes.
channels=2
device = "cpu"
n_neurons = channels * 34 * 34
n_classes = 10

source_layer = Input(n=n_neurons, traces=True)
hidden_layer = LIFNodes(n=n_neurons, traces=True)
target_layer = LIFNodes(n=10, traces=True)

# Connect the two layers.
connection = Connection(
    source=source_layer, target=hidden_layer, update_rule=PostPre, nu=(1e-4, 1e-2)
)

connection2 = Connection(
    source=hidden_layer, target=target_layer, update_rule=PostPre, nu=(1e-4, 1e-2)
)

# recurrent connections
rec_connection = Connection(
    source=hidden_layer, target=hidden_layer, update_rule=NoOp, nu=(1e-4, 1e-2), wmax=0.5, wmin=0.5
)

rec_connection2 = Connection(
    source=target_layer, target=target_layer, update_rule=NoOp, nu=(1e-4, 1e-2), wmax=0.5, wmin=0.5
)

monitor = Monitor(
    obj=target_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.  # Length of simulation (if known ahead of time).
)

network.add_layer(layer=source_layer, name="A")
network.add_layer(layer=hidden_layer, name="B")
network.add_layer(layer=target_layer, name="C")
network.add_connection(connection=connection, source="A", target="B")
network.add_connection(connection=connection2, source="B", target="C")
network.add_connection(connection=rec_connection, source="B", target="B")
network.add_connection(connection=rec_connection2, source="C", target="C")
network.add_monitor(monitor=monitor, name="C")

# Record spikes during the simulation.
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((10, n_classes), device=device)

spike_record = torch.zeros((8, 200, 10), device="cpu")
spikes = {}
spikes['C'] = monitor
print(spike_record.shape)

# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
trainloader, testloader = createMNISTDataloaders(1)

pool = torch.nn.MaxPool2d(1)
accuracy = {"top1": [], "top3": []}
labels = []

# Simulate network on input data.
train_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Top3"])
for epoch in range(10):
    for step, (image, label) in enumerate(trainloader):

        image = pool(image.squeeze())
        image = torch.unsqueeze(image, 1)
        image = image[:200,:,(channels-1), :,:]
        input = {"A": image.reshape([image.size(0), (channels * 34 * 34)])}

        if step % 8 == 0 and step > 0:
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
            top3preds = torch.sort(acc_rates, dim=1, descending=True)[1][:, :3]
            print(step)
            top3acc = 0
            for i in range(len(label_tensor)):
                # print(label_tensor[i], top3preds[i])
                if label_tensor[i] in top3preds[i]:
                    top3acc += 1
            top3acc /= len(label_tensor)

            proportion_pred = proportion_weighting(spikes=spike_record, assignments=assignments,
                                                   proportions=proportions, n_labels=n_classes, )
            # Compute network accuracy according to available classification strategies.
            top1acc = torch.sum(label_tensor.long() == all_activity_pred).item() / len(label_tensor)
            accuracy["top1"].append(100 * top1acc)

            accuracy["top3"].append(100 * top3acc)

            print("\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)" % (
                accuracy["top1"][-1], np.mean(accuracy["top1"]), np.max(accuracy["top1"]),))
            print("Top 3 accuracy: %.2f (last), %.2f (average), %.2f (best)\n" % (
                accuracy["top3"][-1], np.mean(accuracy["top3"]), np.max(accuracy["top3"]),))

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(spikes=spike_record, labels=label_tensor,
                                                            n_labels=n_classes, rates=rates, )

            x = [[epoch, step / 8, top1acc, top3acc, label_tensor.numpy(), top3preds[:, 0]]]
            tmp_df = pd.DataFrame(x, columns=["Epoch", "Iteration", "Accuracy", "Top3", "Prediction", "Target"])
            train_hist = pd.concat([train_hist, tmp_df])

            labels = []

        network.run(inputs=input, time=200)
        labels.append(label)
        spike_record[step % 8] = spikes["C"].get("s").squeeze()
        network.reset_state_variables()
        if step >= 4001:
            labels = []
            break
train_hist.to_csv('train–mnist-2rnn-binds.csv')

epoch = 0
test_hist = pd.DataFrame(columns=["Epoch", "Iteration", "Accuracy", "Top3"])
for step, (image, label) in enumerate(testloader):

    image = pool(image.squeeze())
    image = torch.unsqueeze(image, 1)
    image = image[:200]
    input = {"A": image.reshape([image.size(0), (2 * 34 * 34)])}

    network.run(inputs=input, time=200)
    speik = spikes["C"].get("s")
    speik = speik.squeeze()
    spike_record[step] = speik

    assignments, proportions, rates = assign_labels(
        spikes=spike_record,
        labels=label,
        n_labels=10,
        rates=rates,
    )
    all_activity_pred = all_activity(
        spikes=spike_record,
        assignments=assignments,
        n_labels=10
    )

    label_tensor = torch.tensor(label, device=device)

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
    for i in range(len(label_tensor)):
        # print(label_tensor[i], top3preds[i])
        if label_tensor[i] in top3preds[i]:
            top3acc += 1
    top3acc /= len(label_tensor)
    top1acc = torch.sum(label_tensor.long() == all_activity_pred).item() / len(label_tensor)

    print(all_activity_pred)
    print("Epoch:", epoch, "Step:", step, "Accuracy:",
          100 * torch.sum(label_tensor.long() == all_activity_pred).item() / all_activity_pred.__len__(), label)

    x = [[epoch, step / 8, top1acc, top3acc]]
    tmp_df = pd.DataFrame(x, columns=["Epoch", "Iteration", "Accuracy", "Top3"])
    train_hist = pd.concat([train_hist, tmp_df])

    if step >= 499:
        break

test_hist.to_csv('test–mnist-2rnn-binds.csv')
