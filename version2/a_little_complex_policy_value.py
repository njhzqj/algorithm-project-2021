"""
A neural Network with only one convolutinal block and without residual block
"""
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ValuePolicyNetwork(nn.Module):
    def __init__(self):
        super(ValuePolicyNetwork,self).__init__()
        
        # convolutional block
        self.conv1 = nn.Conv2d(2,256,kernel_size=3,stride=1,padding=1)
        self.batch_normal1 = nn.BatchNorm2d(256)

        # residual block
        self.resi_conv1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.resi_batch_normal1 = nn.BatchNorm2d(256)
        self.resi_conv2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.resi_batch_normal2 = nn.BatchNorm2d(256)

        self.resi_conv3 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.resi_batch_normal3 = nn.BatchNorm2d(256)
        self.resi_conv4 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.resi_batch_normal4 = nn.BatchNorm2d(256)

        self.resi_conv5 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.resi_batch_normal5 = nn.BatchNorm2d(256)
        self.resi_conv6 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.resi_batch_normal6 = nn.BatchNorm2d(256)

        # policy head
        self.policy_conv1 = nn.Conv2d(256,2,kernel_size=1,stride=1,padding=1)
        self.policy_batch_normal = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(in_features=578,out_features=225)
        
        # value head
        self.value_conv1 = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=1)
        self.value_batch_normal = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(289,256)
        self.value_fc2 = nn.Linear(256,1)



    def forward(self,x):
        # convolutional block
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = F.relu(x)


        x1 = x
        # residual block
        r_x = self.resi_conv1(x1)
        r_x = self.resi_batch_normal1(r_x)
        r_x = F.relu(r_x)
        r_x = self.resi_conv2(r_x)
        r_x = self.resi_batch_normal2(r_x)
        x1 = r_x + x1
        x2 = F.relu(x1)

        # residual block
        r_x = self.resi_conv3(x2)
        r_x = self.resi_batch_normal3(r_x)
        r_x = F.relu(r_x)
        r_x = self.resi_conv4(r_x)
        r_x = self.resi_batch_normal4(r_x)
        x2 = r_x + x2
        x3 = F.relu(x2)

        # residual block
        r_x = self.resi_conv5(x3)
        r_x = self.resi_batch_normal5(r_x)
        r_x = F.relu(r_x)
        r_x = self.resi_conv6(r_x)
        r_x = self.resi_batch_normal6(r_x)
        x3 = r_x + x3
        x = F.relu(x3)

        # policy head
        policy_x = self.policy_conv1(x)
        policy_x = self.policy_batch_normal(policy_x)
        policy_x = F.relu(policy_x)
        policy_x = torch.flatten(policy_x, 1)
        # print(policy_x.shape)
        policy_x = self.policy_fc1(policy_x)
        policy_x = F.log_softmax(policy_x,dim=1)

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
        self.optimizer = optim.Adam(self.model.parameters())

    def save_model(self,path):
        torch.save(self.model.state_dict(),path)

    def load_model(self,path):
        net_param = torch.load(path)
        self.model.load_state_dict(net_param)

    def train(self,dataset,act_probs,state_values):
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

    # get one input of input size:
    #test_data = np.random.randint(0,2,(1,2,15,15))
    #test_data[0,1,:] = 0
    #print(test_data)
    # get the output of that input:
    #test_result = vp.test(test_data)
    #print(test_result)

    # test the train function

    
    train_data = np.random.randint(0,3,(5,2,15,15))
    train_data[:2,1,:] = 1
    act_probs = np.random.randint(0,2,(5,225))
    #act_probs = np.random.rand(5,225)
    value = np.random.rand(5)
    
    print(vp.test(train_data,act_probs,value))

    for i in range(30):
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




