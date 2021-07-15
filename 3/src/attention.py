from utils import *
import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import math


class AttentionEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x, neighbor):
        x = x.unsqueeze(2)
        neighbor = neighbor.permute(0, 1, 3, 2)
        agg = x.squeeze(2) + torch.sum(torch.nn.functional.softmax(torch.matmul(x, neighbor) / np.sqrt(self.hidden_dim), dim=-1) * neighbor, dim=-1)
        return agg


class AttentionPointer(nn.Module):
    def __init__(self, hidden_dim, use_tanh=False, use_cuda=False):
        super(AttentionPointer, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.project_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.project_x = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)
        self.C = 10
        self.tanh = nn.Tanh()
        v = torch.FloatTensor(hidden_dim)
        if use_cuda:
            v = v.cuda()
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(hidden_dim)), 1. / math.sqrt(hidden_dim))

    def forward(self, hidden, x):
        x = x.permute(1, 2, 0)
        q = self.project_hidden(hidden).unsqueeze(2)
        e = self.project_x(x)
        expanded_q = q.repeat(1, 1, e.size(2))
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits
