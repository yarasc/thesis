import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate
from snntorch._neurons.neurons import *


class RFFNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, beta,grad):
        super().__init__()

        # initialize layers
        # self.flat = nn.Flatten()

        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.lif1 = snn.RLeaky(beta=beta, spike_grad=grad, V=0.5)
        #V: Recurrent weights to scale output spikes


    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif1.init_leaky()

        mem_rec = []
        spk_rec = []
        spk = x[0]
        for step in range(x.size(0)):
            cur = self.fc1(x[step])
            spk, mem = self.lif1(cur, spk, mem)
            spk_rec.append(spk)
            # mem_rec.append(mem)

        return torch.stack(spk_rec)  # , torch.stack(mem_rec, dim=0)


class TwoFFNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, output=True)

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
            cur2 = self.fc2(spk1)
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
