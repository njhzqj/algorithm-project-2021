"""
A policy value neural network with conv layers.
"""
from numpy.core.fromnumeric import size
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ValuePolicyNetwork(nn.Module):
    def __init__(self):
        super(ValuePolicyNetwork,self).__init__()

        # Can we use drop out to make performance better?
        # self.dropout = nn.Dropout(p=0.2)
        
        # convolutional block 1
        self.conv1 = nn.Conv2d(2,128,kernel_size=3,stride=1,padding=1)
        self.batch_normal1 = nn.BatchNorm2d(128)

        # convolutional block 2
        self.conv2 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.batch_normal2 = nn.BatchNorm2d(128)

        # convolutional block 3
        self.conv3 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.batch_normal3 = nn.BatchNorm2d(128)

        # policy head
        self.policy_conv1 = nn.Conv2d(128,2,kernel_size=1,stride=1,padding=1)
        self.policy_batch_normal = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(in_features=578,out_features=225)
        
        # value head
        self.value_conv1 = nn.Conv2d(128,4,kernel_size=1,stride=1,padding=1)
        self.value_batch_normal = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(1156,256)
        self.value_fc2 = nn.Linear(256,1)



    def forward(self,x):
        # convolutional block 1
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = F.relu(x)

        # convolutional block 2
        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = F.relu(x)

        # convolutional block 3
        x = self.conv3(x)
        x = self.batch_normal3(x)
        x = F.relu(x)

        # policy head
        policy_x = self.policy_conv1(x)
        policy_x = self.policy_batch_normal(policy_x)
        policy_x = F.relu(policy_x)
        policy_x = torch.flatten(policy_x, 1)
        policy_x = self.policy_fc1(policy_x)
        policy_x = F.log_softmax(policy_x,dim=1)

        # print(x.shape)

        # policy head
        value_x = self.value_conv1(x)
        value_x = self.value_batch_normal(value_x)
        value_x = F.relu(value_x)
        value_x = torch.flatten(value_x, 1)
        # print(value_x.shape)
        value_x = self.value_fc1(value_x)
        value_x = F.relu(value_x)
        value_x = self.value_fc2(value_x)
        value_x = torch.tanh(value_x)

        return policy_x,value_x


class ValuePolicy():
    def __init__(self,modelfile=None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ValuePolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=5e-4)
        # self.steps = 0

    def save_model(self,path):
        torch.save(self.model.state_dict(),path)

    def load_model(self,path):
        net_param = torch.load(path)
        self.model.load_state_dict(net_param)

    def train(self,dataset,act_probs,state_values):
        #self.steps += 1
        #if self.steps > 1000:
        #    for g in self.optimizer.param_groups:
        #        g['lr'] = 1e-4


        if self.device == "cuda":
            dataset = Variable(torch.FloatTensor(dataset).cuda())
            act_probs = Variable(torch.FloatTensor(act_probs).cuda())
            state_values = Variable(torch.FloatTensor(state_values).cuda())
        else:
            dataset = Variable(torch.FloatTensor(dataset))
            act_probs = Variable(torch.FloatTensor(act_probs))
            state_values = Variable(torch.FloatTensor(state_values))

        self.optimizer.zero_grad()
        output_act_probs,output_values = self.model(dataset)
        # print(output_act_probs.shape,output_values.shape)
        policy_loss = -torch.mean(torch.sum(act_probs*output_act_probs, 1))
        value_loss = F.mse_loss(output_values.view(-1), state_values)
        loss= value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        return policy_loss.item(),value_loss.item()

    def policy_value(self,dataset):
        if self.device == "cuda":
            dataset = Variable(torch.FloatTensor(dataset).cuda())
            act_probs,values = self.model(dataset)
            return np.exp(act_probs.data.cpu().numpy()),values.data.cpu().numpy()
        else:
            dataset = Variable(torch.FloatTensor(dataset))
            act_probs,values = self.model(dataset)
            return np.exp(act_probs.data.numpy()),values.data.numpy()

    def test(self,dataset,act_probs,state_values):
        if self.device == "cuda":
            dataset = Variable(torch.FloatTensor(dataset).cuda())
            act_probs = Variable(torch.FloatTensor(act_probs).cuda())
            state_values = Variable(torch.FloatTensor(state_values).cuda())
        else:
            dataset = Variable(torch.FloatTensor(dataset))
            act_probs = Variable(torch.FloatTensor(act_probs))
            state_values = Variable(torch.FloatTensor(state_values))

        output_act_probs,output_values = self.model(dataset)

        policy_loss = -torch.mean(torch.sum(act_probs*output_act_probs, 1))
        value_loss = F.mse_loss(output_values.view(-1), state_values)
        return policy_loss.item(),value_loss.item()

if __name__ == "__main__":
    vp = ValuePolicy()
    train_data = np.random.randint(0,3,(5,2,15,15))
    train_data[:2,1,:] = 1
    act_probs = np.random.randint(0,2,(5,225))
    #act_probs = np.random.rand(5,225)
    value = np.random.rand(5)
    
    print(train_data.shape)
    print(act_probs.shape)
    print(value.shape)

    print(vp.test(train_data,act_probs,value))

    for i in range(50):
        loss = vp.train(train_data,act_probs,value)
        print("loss:",loss)

    vp.save_model("test_model_01.pt")

    print("value",value,value.shape)
    print(vp.policy_value(train_data)[1])

    del vp

    vp = ValuePolicy()

    print("value",value)
    print(vp.policy_value(train_data)[1])

    vp.load_model("test_model_01.pt")

    print("value",value)
    print(vp.policy_value(train_data)[1])

    print(vp.test(train_data,act_probs,value))




