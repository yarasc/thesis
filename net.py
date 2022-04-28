import torch
import torch.nn as nn
import snntorch as snn


class RNNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.RNN(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(x.size(0)):
            cur = self.fc1(x)
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc1(x)
            spk2, mem2 = self.lif1(cur, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec)


class FFNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, beta):
        super().__init__()

        # initialize layers
        #self.flat = nn.Flatten()
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.lif1 = snn.Leaky(beta=beta)


    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(x.size(0)):
            cur = self.fc1(x)
            spk, mem = self.lif1(cur, mem)
            spk_rec.append(spk)
            #mem_rec.append(mem)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec, dim=0)

class TwoFFNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(x.size(0)):
            cur = self.fc1(x)
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc1(x)
            spk2, mem2 = self.lif1(cur, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec)
