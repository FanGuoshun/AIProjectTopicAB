import torch

''' 本实验思路为设计不同深度的神经网络解决分类问题，观察不同深度的神经网络对损失函数平滑性的影响 '''
torch.manual_seed(1)

# 制造数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), ).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)
