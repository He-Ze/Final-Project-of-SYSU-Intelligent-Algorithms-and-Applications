from utils import *
from environment import *
from model import *
import tqdm
from torch.nn import CrossEntropyLoss


train_loader, test_loader = read_data()

model = Model(node_hidden_dim, edge_hidden_dim, gcn_num_layers, k).to(device)
lossf = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loss = []

for i in range(num_epochs):
    loss_per_epoch = 0
    for item in tqdm.tqdm(train_loader):
        graph, demand, distance = item[0].to(device), item[1].to(device), item[2].to(device)
        env = Environment(graph, demand, distance)
        sample_logprob, sample_distance, greedy_distance, target_matrix, predict_matrix = model(env)
        predict_matrix = predict_matrix.view(-1, 2)
        target_matrix = target_matrix.view(-1)
        classification_loss = lossf(predict_matrix.to(device), target_matrix.to(device))
        advantage = (sample_distance - greedy_distance).detach()
        reinforce = advantage * sample_logprob
        sequancial_loss = reinforce.sum()
        loss = alpha * sequancial_loss + beta * classification_loss
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss

    train_loss.append(loss_per_epoch)
    print('epoch: %d -train loss: %.4f' %(i, train_loss[-1]))
    write_loss('train_loss.csv', i, train_loss[-1])


# torch.save(model.state_dict(), '../result/params.pkl')
plot_loss(train_loss)