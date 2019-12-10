import torch
import matplotlib.pyplot as plt
import numpy as np
from src.datas import x, y
from src.networks.nn_3layer import Net as Net3

net = Net3(n_feature=2, n_hidden=6, n_output=2)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

loss_x = []
loss_y = []

plt.ion()
plt.figure(1, figsize=(8, 6))

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_x.append(t)
    loss_y.append(loss.data)

    if t % 2 == 0:

        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.subplot(121)
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.text(1.5, -5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})

        plt.pause(0.1)

plt.subplot(122)
print(loss_x)
print(loss_y)
plt.scatter(np.array(loss_x), np.array(loss_y), c='red', label='Loss')

plt.ioff()
plt.show()
