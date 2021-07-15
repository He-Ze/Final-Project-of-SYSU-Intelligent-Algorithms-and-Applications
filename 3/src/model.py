from utils import *
from GCN import *
from decoder import *
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
                 node_hidden_dim,
                 edge_hidden_dim,
                 gcn_num_layers,
                 k):
        super(Model, self).__init__()

        self.GCN = GCN(node_hidden_dim, edge_hidden_dim,
                       gcn_num_layers, k)
        self.sequencialDecoder = SequencialDecoder(node_hidden_dim, use_cuda)
        self.classificationDecoder = ClassificationDecoder(edge_hidden_dim)

    def seqDecoderForward(self, env, h_node, strategy='sample'):
        env.reset()
        last_node = torch.zeros((batch_size, 1)).long().to(device)
        hidden = torch.zeros((2, batch_size, node_hidden_dim)).to(device)
        mask = torch.zeros((batch_size, node_num + 1), dtype=torch.bool).to(device)
        mask[:, 0] = True
        log_prob = 0
        while (env.visited == True).all() == False:
            idx, prob, hidden = self.sequencialDecoder(h_node, last_node, hidden, mask, strategy=strategy)
            env.step(idx)
            last_node = idx
            log_prob = log_prob + torch.log(prob)
            mask = env.mask(mask)

        total_dist, matrix = env.get_state()

        return total_dist, log_prob, matrix

    def forward(self, env):
        x_c = env.graph
        x_d = env.demand
        m = env.distance
        h_node, h_edge = self.GCN(x_c, x_d, m)
        batch_size, node_num, node_hidden_dim = h_node.shape
        sample_distance, sample_logprob, target_matrix = self.seqDecoderForward(env, h_node, strategy='sample')
        greedy_distance, _, _ = self.seqDecoderForward(env, h_node, strategy='greedy')
        predict_matrix = self.classificationDecoder(h_edge)

        return sample_logprob, sample_distance, greedy_distance, target_matrix, predict_matrix
