import csv
import matplotlib.pyplot as plt
import re
import torch

file_path = '../dataset/'
file_name = 'G-20'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
use_cuda = False

node_hidden_dim = 100
edge_hidden_dim = 100
gcn_num_layers = 1

num_epochs = 10
batch_size = 64
beta = 1
learning_rate = 1e-4
weight_decay = 0.96

node_num = 20  # number of customers
initial_capacity = 5  # initial capacity of vehicles
k = 10  # number of nearest neighbors
alpha = 1


def write_loss(file_name, epoch, loss):
    file_path = '../result/' + file_name

    if epoch == 0:
        with open(file_path, "w+", newline='\n') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([epoch, float(loss)])
    else:
        with open(file_path, "a+", newline='\n') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([epoch, float(loss)])


def plot_loss(loss):
    file_path = '../result/' + 'loss.png'
    plt.plot(loss, color='skyblue', linewidth=1)
    plt.title('Train Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(file_path)
