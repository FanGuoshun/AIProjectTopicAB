import torch
import torch.nn.functional as F


# 3层神经网络，输入层加两层隐藏层再加一层输出层
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # 使用RelU作为激活函数
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)
