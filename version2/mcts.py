"""
This file contains the mcts algothrim and classes and functions which are needed.
"""
from Gomoku import *
from a_little_complex_policy_value import *
import math
import copy
import time
import random

TIMELIMIT = 6 # 6 seconds for one mcts seacher to do one search
SIMULATIONTIMESLIMIT = 2000 # max 1000 simulations for one mcts search for one step

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
        if self.grid[act[0]][act[1]] != SPACE:
            return None
        new_node = StateNode(self.grid,player)
        new_node.grid[act[0]][act[1]] = player
        new_node.parent = self
        new_node.player = getOponent(self.player)
        new_node.act = act
        self.children.append(new_node)
        return new_node

    def show(self):
        print("\ncurrent node:")
        print("player: ",self.player)
        print("parent: ",self.parent,"child num",len(self.children))
        print("action: ",self.act)
        print("N(parent,a) (visit_count",self.visit_count)
        print("W(parent,a) (total_action_value",self.total_action_value)
        print("Q(parent,a) (mean_action_value",self.mean_action_value)
        print("P(parent,a) (prior_probability",self.prior_probability)
        print()

class MctsSeacher:

    def __init__(self,nn_model) -> None:
        self.root = None
        self.nn_model = nn_model
        self.tau = 0.25

    def mctsSearch(self,grid,player):
        start = time.time()

        self.player = player
        self.root = StateNode(grid,self.player)

        self.root.show()

        for i in range(SIMULATIONTIMESLIMIT):
            if time.time() - start > TIMELIMIT:
                break
            leaf_node = self.select()
            reward = self.simulation(leaf_node)
            self.backup(leaf_node,reward)

        act = self.bestAction(self.root)
        print("decision time:",time.time()-start," total simulation times: ",i)
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
        random.shuffle(actions_list)
        for action,probility in actions_list:
            new_node = node.newNode(action,node.player)
            if new_node != None:
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
            if isWinAfterPlace(self.root.grid,child.act[0],child.act[1],child.player):
                return child.act
            if isWinAfterPlace(self.root.grid,child.act[0],child.act[1],getOponent(child.player)):
                return child.act
            score = child.visit_count  / node.visit_count 
            print(child.act,score,child.visit_count)
            if index == 0:
                max_score = score
                max_index = index
            else:
                if score > max_score:
                    max_score = score
                    max_index = index
        print(node.children[max_index].prior_probability)
        return node.children[max_index].act
        # 实际上这里有一个论文中十分重要的优化没有实现，即在整局游戏中维护同一棵树，只是不断地调整根节点
        # 而不是每次重新模拟一棵树
