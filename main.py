import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self,hidden_size,output_size,input_size):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.l3 = nn.Linear(hidden_size,hidden_size)
        self.relu3 = nn.LeakyReLU()
        self.l4 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        x = self.l4(x)
        return x

criterion = nn.MSELoss()

def initial_comdition_loss(y,target_value):
    return nn.MSELoss(y,target_value)

t_numpy = np.arange(0,5+0.01,0.01,dtype=np.float32)
t = torch.from_numpy(t_numpy).reshape(len(t_numpy),1)
t.requires_grad = True
k = 1

model = NeuralNet(hidden_size=50)

learning_rate = 8e-3
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

num_epochs = 10000

for epoch in range(num_epochs):
     epsilon = torch.normal(0,0.1,size=(len(t),1)).float()
     t_train = t+epsilon
     y_pred = model(t_train)
     dy_dt = torch.autograd.grad