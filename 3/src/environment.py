from utils import *
from model import *
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset


def loadd(data):
    graph, demand, distance = (data[i] for i in data.files)
    dataset = TensorDataset(torch.FloatTensor(graph), torch.FloatTensor(demand), torch.FloatTensor(distance))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


def read_data():
    train_loader = loadd(np.load(file_path + 'data/' + file_name + '-training.npz'))
    test_loader = loadd(np.load(file_path + 'data/' + file_name + '-testing.npz'))
    return train_loader, test_loader

class Environment:
    def __init__(self, graph, demand, distance):
        self.batch_size = batch_size
        self.node_num = node_num
        self.initial_capacity = initial_capacity
        self.k = k
        self.graph = graph
        self.demand = demand
        self.distance = distance

        self.visited, self.routes, self.remaining_capacity, self.remaining_demands = self.init()
        self.steptimes = 0

    def init(self):
        visited = torch.zeros(self.batch_size, self.node_num+1, dtype=torch.bool)
        visited[:, 0] = True
        routes = torch.full((self.batch_size, 1), 0, dtype=torch.long)
        remaining_capacity = torch.full(size=(self.batch_size, 1), fill_value=self.initial_capacity, dtype=torch.float)
        remaining_demands = self.demand.clone().float()
        return visited.to(device), routes.to(device), remaining_capacity.to(device), remaining_demands.to(device)

    def reset(self):
        self.visited, self.routes, self.remaining_capacity, self.remaining_demands = self.init()
        self.steptimes = 0

    def step(self, action):
        action = action.squeeze(-1)
        self.visited.scatter_(1, action.unsqueeze(1), True)
        self.routes = torch.cat((self.routes, action.unsqueeze(1)), dim=1)
        prev_capacity = self.remaining_capacity
        curr_demands = self.remaining_demands.gather(1, action.unsqueeze(1))
        self.remaining_capacity[action==0] = self.initial_capacity
        self.remaining_capacity[action!=0] = torch.maximum(torch.zeros(self.batch_size, 1).to(device), prev_capacity[action!=0] - curr_demands[action!=0])
        self.remaining_demands.scatter_(1, action.unsqueeze(1), 0)
        self.steptimes = self.steptimes + 1

    def mask(self, last_mask):
        mask = last_mask.clone()
        mask[self.remaining_demands==0] = True
        mask[self.remaining_demands>self.remaining_capacity] = True
        if self.steptimes == 1:
            mask[:, 0] = True
        mask[self.routes[:, -2]==0, 0] = True
        mask[mask.all(dim=1), 0] = False
        return mask

    def dist_step(self, prev_step, curr_step):
        idx = torch.arange(start=0, end=batch_size, step=1).unsqueeze(1)
        reward = self.distance[idx, prev_step, curr_step]
        return reward

    def get_reward(self):
        prev_step = self.routes[:, -2:-1]
        curr_step = self.routes[:, -1:]
        reward = self.dist_step(prev_step, curr_step)
        return reward

    def get_state(self):
        total_dist = torch.zeros(self.batch_size, 1).to(device)
        for i in range(1, self.routes.size(-1)):
            prev_step = self.routes[:, (i - 1):i]
            curr_step = self.routes[:, i:(i + 1)]
            dist = self.dist_step(prev_step, curr_step)
            total_dist = total_dist + dist

        matrix = torch.zeros(self.batch_size, self.node_num + 1, self.node_num + 1, dtype=torch.float)
        idx = torch.arange(start=0, end=batch_size, step=1).unsqueeze(1)
        for i in range(1, self.routes.size(-1)):
            prev_step = self.routes[:, (i - 1):i]
            curr_step = self.routes[:, i:(i + 1)]
            matrix[idx, prev_step, curr_step] = 1

        return total_dist, matrix.long()