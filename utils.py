import time

import pandas as pd
import tonic
import tonic.transforms as transforms
from snntorch import functional as SF
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def createDataloaders(batch_size):
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size,
                           time_window=3000)])

    trainset = tonic.datasets.DVSGesture(save_to='./dvs_data', transform=frame_transform, train=True)
    testset = tonic.datasets.DVSGesture(save_to='./dvs_data', transform=frame_transform, train=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

    return trainloader, testloader


def dataloaderWithSpecTransform(batch_size, transform):
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    if transform == "averaged timesurface":
        frame_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToAveragedTimesurface(sensor_size=sensor_size,
                                             cell_size=10,
                                             surface_size=7,
                                             temporal_window=200.0,
                                             tau=5000.0,
                                             decay='lin',
                                             )
        ])
    elif transform == "timesurface":
        frame_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToTimesurface(sensor_size=sensor_size)
        ])
    elif transform == "voxelgrid":
        frame_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToAveragedTimesurface(sensor_size=sensor_size)
        ])
    else:
        frame_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size,
                               time_window=3000)
        ])
    print("begin")
    trainset = tonic.datasets.DVSGesture(save_to='./dvs_data/events', transform=frame_transform, train=True)
    print("testset")
    testset = tonic.datasets.DVSGesture(save_to='./dvs_data/events', transform=frame_transform, train=False)

    print("maketrainloaders")
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    print("return")
    return trainloader, testloader


def createMNISTDataloaders(batch_size):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                          transforms.ToFrame(sensor_size=sensor_size,
                                                             time_window=1000)
                                          ])
    trainset = tonic.datasets.NMNIST(save_to='./dvs_data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./dvs_data', transform=frame_transform, train=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
    return trainloader, testloader


def evaluate(spk_rec, targets, t0, loss_val, epoch, i, train):

    # print(spk_rec.sum(dim=0))
    batch_nr = 0
    top3 = 0

    for pred in spk_rec.sum(dim=0):
        _, idx3 = pred.topk(3)
        # print(targets[batch_nr], idx1[batch_nr], idx3)
        if targets[batch_nr] in idx3:
            top3 += 1
        batch_nr += 1
    top3 /= batch_nr

    acc = SF.accuracy_rate(spk_rec, targets)
    # Store loss history for future plotting every 0.3 sec
    # _, idx = spk_rec.sum(dim=0).max(3)

    x = [[epoch, i, acc, loss_val.item(), top3]]
    tmp_df = pd.DataFrame(x, columns=["Epoch", "Iteration", "Accuracy", "Loss", "Top3"])

    print('{} s'.format(time.time() - t0), end=": ")

    if train:
        print(f"Epoch {epoch}, Iteration {i} – Train Loss: {loss_val.item():.2f}", end=" – ")
    else:
        print(f"Epoch {epoch}, Iteration {i} – Test Loss: {loss_val.item():.2f}", end=" – ")
    print(f"Accuracy: {acc * 100:.2f}%", SF.accuracy_rate(spk_rec, targets))

    return tmp_df

def evaluate10(spk_rec, targets, t0, loss_val, epoch, i, train):

    # print(spk_rec.sum(dim=0))
    batch_nr = 0
    top3 = 0
    top1 = 0
    other = False
    for pred in spk_rec.sum(dim=0):
        _, idx1 = pred.topk(1)
        _, idx3 = pred.topk(3)
        # print(targets[batch_nr], idx1[batch_nr], idx3)
        if targets[batch_nr] < 10 or other:
            if targets[batch_nr] in idx3:
                top3 += 1
            if targets[batch_nr] == idx1:
                top1 += 1
            batch_nr += 1
    top3 /= batch_nr
    top1 /= batch_nr

    #acc = SF.accuracy_rate(spk_rec, targets)
    # Store loss history for future plotting every 0.3 sec
    # _, idx = spk_rec.sum(dim=0).max(3)

    x = [[epoch, i, top1, loss_val.item(), top3]]
    tmp_df = pd.DataFrame(x, columns=["Epoch", "Iteration", "Accuracy", "Loss", "Top3"])

    print('{} s'.format(time.time() - t0), end=": ")

    if train:
        print(f"Epoch {epoch}, Iteration {i} – Train Loss: {loss_val.item():.2f}", end=" – ")
    else:
        print(f"Epoch {epoch}, Iteration {i} – Test Loss: {loss_val.item():.2f}", end=" – ")
    print(f"Accuracy: {top1 * 100:.2f}%", SF.accuracy_rate(spk_rec, targets))

    return tmp_df


def visualize():
    pass
    # x = np.arange(num_epochs)
    # fig = plt.figure(facecolor="w")
    # plt.errorbar(y=test_hist.accuracy, x=x, yerr=test_hist.error_acc, label='Test/Val')
    # plt.errorbar(y=train_hist.accuracy, x=x, yerr=train_hist.error_acc, label='Train')
    # plt.title("Average Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend(loc='lower right')
    # plt.show()
    # # plt.savefig("accuracy.png", dpi=150)
    #
    #
    # fig = plt.figure(facecolor="w")
    # plt.errorbar(y=test_hist.loss, x=x, yerr=test_hist.error_los, label='Test/Val')
    # plt.errorbar(y=train_hist.loss, x=x, yerr=train_hist.error_loss, label='Train')
    # plt.title("Average Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend(loc='lower right')
    # plt.show()
    # # plt.savefig("loss.png", dpi=150)


class ce_rate_loss_10(SF.LossFunctions):
    def __init__(self):
        self.__name__ = "ce_rate_loss"

    def __call__(self, spk_out, targets):
        device, num_steps, _ = self._prediction_check(spk_out)
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss(ignore_index=10)

        log_p_y = log_softmax_fn(spk_out)
        loss = torch.zeros((1), dtype=torch.float, device=device)

        for step in range(num_steps):
            loss += loss_fn(log_p_y[step], targets)

        return loss / num_steps

class ce_rate_loss_8(SF.LossFunctions):
    def __init__(self):
        self.__name__ = "ce_rate_loss"

    def __call__(self, spk_out, targets):
        device, num_steps, _ = self._prediction_check(spk_out)
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss(ignore_index=10)

        log_p_y = log_softmax_fn(spk_out)
        loss = torch.zeros((1), dtype=torch.float, device=device)

        for i in range(len(targets)):
            if targets[i] == 4:
                targets[i] = 3
            elif targets[i] == 6:
                targets[i] = 5

        for step in range(num_steps):
            loss += loss_fn(log_p_y[step], targets)

        return loss / num_steps
