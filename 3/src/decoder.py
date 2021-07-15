from utils import *
from attention import *
import torch
import torch.nn as nn

class SequencialDecoder(nn.Module):
    def __init__(self, hidden_dim, use_cuda=False):
        super(SequencialDecoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2)
        self.tanh = nn.Tanh()
        self.h = nn.Linear(hidden_dim, 1)
        self.W = nn.Linear(2, 1)
        self.pointer = AttentionPointer(hidden_dim, use_tanh=True, use_cuda=use_cuda)

    def forward(self, x, last_node, hidden, mask, strategy='sample'):
        batch_size = x.size(0)
        batch_idx = torch.arange(start=0, end=batch_size).unsqueeze(1)
        if use_cuda:
            batch_idx = batch_idx.to(device)
        last_x = x[batch_idx, last_node].permute(1, 0, 2)
        _, hidden = self.gru(last_x, hidden)
        z = hidden[-1]
        # Eq 15
        _, u = self.pointer(z, x.permute(1, 0, 2))
        # Eq 16
        u = u.masked_fill_(mask, -np.inf)
        probs = self.softmax(u)
        if strategy == 'sample':
            # SampleRollout
            idx = torch.multinomial(probs, num_samples=1)
        elif strategy == 'greedy':
            # GreedyRollout
            idx = torch.max(probs, dim=1)[1].unsqueeze(1)
        prob = probs[batch_idx, idx].squeeze(1)

        return idx, prob, hidden

class ClassificationDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassificationDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.softmax = nn.Softmax(-1)

    def forward(self, e):
        a = self.MLP(e)
        a = a.squeeze(-1)
        out = self.softmax(a)
        return out