import torch
import torch.nn.functional as F


# 3层神经网络，输入层加一层隐藏层再加一层输出层
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))  # 使用RelU作为激活函数
        x = self.out(x)
        return x

