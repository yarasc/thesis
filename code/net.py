import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F


class RFFNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, beta, grad):
        super().__init__()

        # initialize layers
        # self.flat = nn.Flatten()

        self.fc1 = nn.Linear(num_inputs, num_inputs)
        self.lif1 = snn.RLeaky(beta=beta, spike_grad=grad, V=0.5)
        # V: Recurrent weights to scale output spikes

    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif1.init_leaky()

        mem_rec = []
        spk_rec = []
        spk = x[0]
        avg_spikes = 0
        for step in range(x.size(0)):
            avg_spikes += x[step].sum()
            cur = self.fc1(x[step])
            spk, mem = self.lif1(cur, spk, mem)
            spk_rec.append(spk)
            # mem_rec.append(mem)
        #print(avg_spikes/x.size(0))
        #TODO print sparcity

        return torch.stack(spk_rec)  # , torch.stack(mem_rec, dim=0)

class TwoRFFNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, grad):
        super().__init__()

        # initialize layers
        # self.flat = nn.Flatten()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.RLeaky(beta=beta, spike_grad=grad, V=0.5)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.RLeaky(beta=beta, spike_grad=grad, V=0.5)
        # V: Recurrent weights to scale output spikes

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif1.init_leaky()

        mem_rec = []
        spk_rec = []
        spk1 = x[0]
        avg_spikes = 0
        for step in range(x.size(0)):
            avg_spikes += x[step].sum()
            cur = self.fc1(x[step])
            spk1, mem = self.lif1(cur, spk1, mem1)
            cur = self.fc2(spk1)
            if step ==0 :
                spk2 = cur
            spk2, mem = self.lif2(cur, spk2, mem2)
            spk_rec.append(spk2)
            # mem_rec.append(mem)
        #print(avg_spikes/x.size(0))

        return torch.stack(spk_rec)  # , torch.stack(mem_rec, dim=0)


class TwoFFNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, grad):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=grad)
        # self.pool = nn.MaxPool2d(kernel_size=3)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, output=True, spike_grad=grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(x.size(0)):
            cur = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur, mem1)

            cur2 = F.max_pool2d(self.fc2(spk1), 2)
            # F.max_pool2d(self.conv2(spk1), 2)
            # cur3 = self.pool(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        return torch.stack(spk_rec)


class FFNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, beta, grad):
        super().__init__()
        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=grad)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []
        for step in range(x.size(0)):
            cur = self.fc1(x[step])
            spk, mem = self.lif1(cur, mem)

            spk_rec.append(spk)
            # mem_rec.append(mem)

        return torch.stack(spk_rec)  # , torch.stack(mem_rec, dim=0)


class CNNet(nn.Module):
    def __init__(self, num_outputs, beta, grad, batch_size, filter, kernel, lin_size):
        super().__init__()
        self.batch_size = batch_size

        # initialize layers
        # self.flat = nn.Flatten()

        self.conv1 = nn.Conv2d(2, filter, (kernel, kernel))
        self.lif1 = snn.RLeaky(beta=beta, spike_grad=grad, V=0.5)
        self.fc1 = nn.Linear(lin_size*filter, num_outputs)
        self.lif2 = snn.RLeaky(beta=beta, spike_grad=grad, V=0.5)

        # V: Recurrent weights to scale output spikes

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        mem_rec = []
        spk_rec = []
        spk1 = x[0]
        for step in range(x.size(0)):
            cur = F.max_pool2d(self.conv1(x[step]), 2)
            spk1, mem1 = self.lif1(cur, mem1, spk1)
            cur = self.fc1(spk1.view(self.batch_size, -1))
            if step == 0:
                spk2 = cur
            spk2, mem2 = self.lif1(cur, mem2, spk2)
            spk_rec.append(spk2)
            # mem_rec.append(mem)

        return torch.stack(spk_rec)  # , torch.stack(mem_rec, dim=0)
