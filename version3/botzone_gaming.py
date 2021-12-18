import numpy as np

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
        net_param = torch.load(path,map_location=torch.device('cpu'))
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

GRIDSIZE = 15

WINCON = 5

BLACK_PLAYER = 0

WHITE_PLAYER = 1

SPACE = 2

def inGrid(x,y):
    """
    Input x, y are in the form of int numbers, which replace a position in the grid.
    This function return True if given (x, y) is in the grid, else False.
    """
    if x < 0 or y < 0 or x >= GRIDSIZE or y >= GRIDSIZE:
        return False
    else:
        return True

def getOponent(player):
    """
    Return the Oponent of given player.
    """
    if player == BLACK_PLAYER:
        return WHITE_PLAYER
    elif player == WHITE_PLAYER:
        return BLACK_PLAYER
    else:
        raise ValueError("A player shall be either a white player or a black player!")

def gridToStr(grid):
    """
    Return the string representation of the given grid, which can be prettily printed in the command line.
    """
    string_grid = ""
    for i in range(GRIDSIZE+2):
        string_grid += "#"
    for i in range(GRIDSIZE):
        string_grid += "\n#"
        for j in range(GRIDSIZE):
            if grid[i][j] == SPACE:
                string_grid += " "
            else:
                string_grid += str(grid[i][j])
        string_grid += "#"
    string_grid += "\n"
    for i in range(GRIDSIZE+2):
        string_grid += "#"
    return string_grid

def checkTowards(grid,x,y,xx,yy,player):
    """
    A function which can be used to check the current state of grid.
    This fucntion start from the given (x,y) posistion of the grid and check both sides of the direction
    of xx,yy and return how many pieces are the same as the given player.
    """
    if not inGrid(x,y):
        raise ValueError("This position is not in the grid!")
    if not player in [BLACK_PLAYER,WHITE_PLAYER]:
        raise ValueError("A player shall be either a white player or a black player!")
    for i in range(1, WINCON + 1):
        temp_x = x + xx * i
        temp_y = y + yy * i
        if (not inGrid(temp_x, temp_y)) or grid[temp_x][temp_y] != player:
            break
    for j in range(1, WINCON + 1):
        temp_x = x - xx * j
        temp_y = y - yy * j
        if (not inGrid(temp_x, temp_y)) or grid[temp_x][temp_y] != player:
            break
    # print(xx,yy,i,j,temp_x,temp_y,i + j - 1)
    return i + j - 1

def isWinAfterPlace(grid,x,y,player):
    """
    Check the grid if after player place a piece in (x,y) he can win the game.
    """
    return checkTowards(grid,x,y,1,0,player) >= 5 or checkTowards(grid,x,y,0,1,player) >= 5 or checkTowards(grid,x,y,1,1,player) >= 5 or checkTowards(grid,x,y,1,-1,player) >= 5
    

def isGridFull(grid):
    """
    If the given grid is full of pieces, return True, else False.
    """
    for i in range(GRIDSIZE):
        for j in range(GRIDSIZE):
            if grid[i][j] == SPACE:
                return False
    return True

def isGridEmpty(grid):
    """
    If there is no piece in the grid, return True, else False.
    """
    for i in range(GRIDSIZE):
        for j in range(GRIDSIZE):
            if grid[i][j] != SPACE:
                return False
    return True

def initEmptyGrid():
    """
    Return a empty a grid.
    """
    return [[SPACE for _ in range(GRIDSIZE)] for _ in range(GRIDSIZE)]

def gridReshape(grid):
    """
    reshape the given grid which is (GRIDSIZE,GRIDSIZE) to a numpy array of (1,GRIDSIZE*GRIDSIZE) 
    """
    return np.reshape(np.array(grid),(1,GRIDSIZE*GRIDSIZE))

def numpyToGrid(nparray):
    """
    reshape the given numpy array of (1,GRIDSIZE*GRIDSIZE) to a grid which is (GRIDSIZE,GRIDSIZE)
    """
    return np.reshape(nparray,(GRIDSIZE,GRIDSIZE)).tolist()

def prob_index_transform(index):
    """
    """
    return int(index/GRIDSIZE),index%GRIDSIZE

import math
import copy

SIMULATIONTIMESLIMIT = 30 # max 1000 simulations for one mcts search for one step

C_PUCT = 5

class StateNode:

    def __init__(self,grid,player) -> None:
        self.grid = copy.deepcopy(grid)
        self.children = []
        self.parent = None
        self.act = None
        self.player = player
        
        self.visit_count = 0 # N(s,a)
        self.total_action_value = 0 # W(s,a)
        self.mean_action_value = 0 # Q(s,a)
        self.prior_probability = 0 # P(s,a)
        # s means the parent of current state node
        # a means current node's action

    def isLeaf(self):
        if len(self.children)==0:
            return True
        else:
            return False
        
    def newNode(self,act,player) -> "StateNode":
        new_node = StateNode(self.grid,player)
        new_node.grid[act[0]][act[1]] = player
        new_node.parent = self
        new_node.player = getOponent(self.player)
        new_node.act = act
        self.children.append(new_node)
        return new_node


class MctsSeacher:

    def __init__(self,nn_model) -> None:
        self.root = None
        self.nn_model = nn_model
        self.tau = 0.25

    def mctsSearch(self,grid,player):
        self.player = player
        self.root = StateNode(grid,self.player)

        for _ in range(SIMULATIONTIMESLIMIT):
            leaf_node = self.select()
            reward = self.simulation(leaf_node)
            self.backup(leaf_node,reward)

        act = self.bestAction(self.root)
        return act

    def select(self):
        temp_node = self.root
        while not temp_node.isLeaf():
            total_visit_times = 0
            for child in temp_node.children:
                total_visit_times += child.visit_count
            for index,child in enumerate(temp_node.children):
                score = child.mean_action_value + C_PUCT * child.prior_probability * math.sqrt(total_visit_times) / (1 + child.visit_count)
                if index == 0:
                    max_score = score
                    max_index = 0
                else:
                    if score > max_score:
                        max_score = score
                        max_index = index
            temp_node = temp_node.children[max_index]
        return temp_node

    def simulation(self,node:StateNode):
        input = np.zeros((1,2,GRIDSIZE,GRIDSIZE))
        input[:,1,:,:] = node.player
        for x in range(GRIDSIZE):
            for y in range(GRIDSIZE):
                input[:,0,x,y] = node.grid[x][y]
        
        actions_probs,value = self.nn_model.policy_value(input)
        actions_probs = actions_probs.reshape(-1)
        actions_index = np.argsort((-actions_probs))[:5]
        action_probilities = -np.sort((-actions_probs))[:5]
        actions_list = [[prob_index_transform(actions_index[i]),action_probilities[i]] for i in range(5)]
        for action,probility in actions_list:
            new_node = node.newNode(action,node.player)
            new_node.prior_probability = probility
        return value[0][0]

    def backup(self,node:StateNode,reward):
        node.visit_count += 1
        node.total_action_value += reward
        node.mean_action_value = node.total_action_value / node.visit_count
        temp_node = node
        while temp_node.parent != None:
            #print("backup,tempnode",temp_node)
            temp_node.parent.visit_count += 1
            temp_node.parent.total_action_value += reward
            temp_node.parent.mean_action_value = temp_node.parent.total_action_value / temp_node.parent.visit_count
            temp_node = temp_node.parent
        

    def bestAction(self,node:StateNode):
        for index,child in enumerate(node.children):
            # score = child.visit_count ** (1 / self.tau) / node.visit_count  ** (1 / self.tau)
            score = child.visit_count  / node.visit_count 
            # print(child.act,score,child.visit_count)
            if index == 0:
                max_score = score
                max_index = index
            else:
                if score > max_score:
                    max_score = score
                    max_index = index
        # print(node.children[max_index].prior_probability)
        return node.children[max_index].act

def jsonToAct(jsonText):
    return (int(jsonText["x"]),int(jsonText["y"]))

def onlineGame():
    import json

    vp = ValuePolicy()
    vp.load_model("/data/gomoku/abc_policy_value_e_1.pt")
    mctsSeacher = MctsSeacher(vp)
    grid = [[SPACE for _ in range(GRIDSIZE)] for _ in range(GRIDSIZE)]
    
    full_input = json.loads(input())
    if "data" in full_input:
        my_data = full_input["data"]
    else:
        my_data = None

    all_requests = full_input["requests"]
    all_responses = full_input["responses"]

    request_size = len(all_requests)

    if int(all_requests[0]["x"]) == -1:
        isBlack = True
        player = BLACK_PLAYER
    else:
        isBlack = False
        player = WHITE_PLAYER
    
    for i in range(request_size):
        if (not isBlack) or i > 0:
            x = int(all_requests[i]["x"])
            y = int(all_requests[i]["y"])
            grid[x][y] = getOponent(player)
        if (i == request_size -1):
            break
        x = int(all_responses[i]["x"])
        y = int(all_responses[i]["y"])
        grid[x][y] = player

    act = mctsSeacher.mctsSearch(grid,getOponent(player))

    my_action = {"x":int(act[0]),"y":int(act[1])}

    print(json.dumps({
        "response": my_action,
        "data": my_data
    }))


if __name__ == "__main__":
    onlineGame()