import torch

torch.manual_seed(1)

# 样本量10，
# n_data = torch.ones(10, 2)
# x0 = torch.normal(2 * n_data, 1)
# y0 = torch.zeros(10)
# x1 = torch.normal(-2 * n_data, 1)
# y1 = torch.ones(10)

# 样本量100
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

# 样本量1000
# n_data = torch.ones(10000, 2)
# x0 = torch.normal(2 * n_data, 1)
# y0 = torch.zeros(1000)
# x1 = torch.normal(-2 * n_data, 1)
# y1 = torch.ones(1000)

x = torch.cat((x0, x1), ).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)
