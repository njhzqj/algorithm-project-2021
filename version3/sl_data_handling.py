"""
this file contains methods and functions that can be used in loading data from dataset
"""

import json
import numpy as np
import copy
import random

BLACK = 0
WHITE = 1
EMPTY = 2
SIZE = 15

hd = []

def getOp(player:int):
    if player == BLACK:
        return WHITE
    elif player == WHITE:
        return BLACK
    else:
        raise ValueError

def getActProbs(x,y):
    grid = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
    grid[x][y] = 1
    return grid

def flip_numpy_state_and_act(state_sets,act_probs,values):
    state_sets_flip0 = np.flip(state_sets,axis=2)
    act_probs_flip0 = np.reshape(np.flip(np.reshape(act_probs,(state_sets.shape[0],SIZE,SIZE)),axis=1),(state_sets.shape[0],SIZE*SIZE))
    state_sets_flip1 = np.flip(state_sets,axis=3)
    act_probs_flip1 = np.reshape(np.flip(np.reshape(act_probs,(state_sets.shape[0],SIZE,SIZE)),axis=2),(state_sets.shape[0],SIZE*SIZE))

    state_set_flip = np.concatenate((state_sets_flip0,state_sets_flip1),axis = 0)
    act_probs_flip = np.concatenate((act_probs_flip0,act_probs_flip1),axis = 0)
    act_probs_values = np.concatenate((values,values),axis = 0)
    return state_set_flip,act_probs_flip,act_probs_values

def rotate_numpy_state_and_act(state_sets,act_probs,values):
    state_set_rota = state_sets
    act_probs_rota = act_probs
    values_rota = values
    for angle in [1,2,3]:
        c_state_set_rota = np.rot90(state_sets,angle,(2,3))
        c_act_probs_rota = np.reshape(np.rot90(np.reshape(act_probs,(state_sets.shape[0],SIZE,SIZE)),angle,(1,2)),(state_sets.shape[0],SIZE*SIZE))
        state_set_rota = np.concatenate((state_set_rota,c_state_set_rota),axis = 0)
        act_probs_rota = np.concatenate((act_probs_rota,c_act_probs_rota),axis = 0)
        values_rota = np.concatenate((values_rota,values),axis = 0)
    return state_set_rota,act_probs_rota,values_rota


    

class GomokuGame:
    def __init__(self) -> None:
        self.player = WHITE
        self.grid = [[EMPTY for i in range(SIZE)] for _ in range(SIZE)]
        self.states_sets = []
        self.act_probs = []
    
    def placePiece(self,x,y,winner:None) -> None:
        self.player = getOp(self.player)
        current_state = copy.deepcopy(self.grid)
        current_act_prob = getActProbs(x,y)
        self.states_sets.append([current_state,[[self.player for _ in range(SIZE)] for _ in range(SIZE)]])
        self.act_probs.append(current_act_prob)

        self.grid[x][y] = self.player

        if winner != None:
            self.values = [0 for _ in range(len(self.states_sets))]
            length = len(self.states_sets) - 1
            if winner == self.player:
                while length >= 0:
                    self.values[length] = 1
                    length -= 2
            else:
                while length >= 0:
                    self.values[length-1] = 1
                    length -= 2
                
            return self.save()
        
    
    def save(self):
        states_sets = np.array(self.states_sets)
        act_probs = np.reshape(np.array(self.act_probs),(len(self.act_probs),SIZE*SIZE))
        values = np.array(self.values)

        states_sets_f,act_probs_f,values_f = flip_numpy_state_and_act(states_sets,act_probs,values)
        states_sets_r,act_probs_r,values_r = rotate_numpy_state_and_act(states_sets,act_probs,values)
        
        return np.concatenate((states_sets_f,states_sets_r),axis=0),np.concatenate((act_probs_f,act_probs_r),axis=0),np.concatenate((values_f,values_r),axis=0)


    def str_repr(self):
        """
        store current game info into a str for command line viewing 
        """
        repr = "\n####CURRENT GRID####\n"
        for i in range(SIZE+2):
            repr += "#"
        repr += "\n"
        for i in range(SIZE):
            repr += "#"
            for j in range(SIZE):
                if self.grid[i][j] == EMPTY:
                    repr += " "
                else:
                    repr += str(self.grid[i][j])
            repr += "#\n"
        for i in range(SIZE+2):
            repr += "#"
        repr += "\n"
        return repr

def decodeOneMatch(json_text:str):

    player_info = json.loads(json_text)['players']
    if player_info[0]["type"] == "bot" and player_info[1]["type"] == "bot":
        if (player_info[0]["bot"] in hd) and (player_info[1]["bot"] in hd):
            log_info = json.loads(json_text)['log']
            c_game = GomokuGame()
            for i in range(len(log_info)):
                if (i%2) == 0 and i != 0:
                    try:
                        error = log_info[i]['output']['display']['error']
                        print("ERROR:",error)
                        return None
                    except KeyError:
                        x = log_info[i]['output']['display']['x']
                        y = log_info[i]['output']['display']['y']
                        try:
                            winner = log_info[i]['output']['display']['winner']
                        except KeyError:
                            winner = None
                        result = c_game.placePiece(x,y,winner)
            return result
    return None

def main():

    total_game_num = 0

    with open("hd.txt","r") as f:
        for line in f:
            hd.append(str(line.strip('\n')))
    # for month in ['5','6','7','8','9','10','11']:
    for game_num in range(200):
        start = game_num * 100 + 1
        end = start + 100 - 1
        file_name = str(start) + "-" + str(end)
        with open("datas\\botzone_data\\raw_data\\Gomoku-2021-5\\"+file_name+'.matches','r') as f:
            flag = True
            states_sets = None
            act_probs = None
            values = None
            for line in f:
                reuslt = decodeOneMatch(line)
                if reuslt != None:
                    total_game_num += 1
                    c_states_sets,c_act_probs,c_values = reuslt
                    if flag:
                        flag = False
                        states_sets = c_states_sets
                        act_probs = c_act_probs
                        values = c_values
                    else:
                        states_sets = np.concatenate((states_sets,c_states_sets),axis=0)
                        act_probs = np.concatenate((act_probs,c_act_probs),axis=0)
                        values = np.concatenate((values,c_values),axis=0)
            print(states_sets.shape,act_probs.shape,values.shape)

            if(states_sets.shape[0]>10):
                indexs = [i for i in range(states_sets.shape[0])]
                random.shuffle(indexs)
                states_sets = states_sets[indexs]
                act_probs = act_probs[indexs]
                values = values[indexs]
                np.save("datas\\botzone_data\\processed_data\\Gomoku-2021-5\\"+file_name+"_states_sets.npy",states_sets)
                np.save("datas\\botzone_data\\processed_data\\Gomoku-2021-5\\"+file_name+"_act_probs.npy",act_probs)
                np.save("datas\\botzone_data\\processed_data\\Gomoku-2021-5\\"+file_name+"_values.npy",values)

    print("total processed game number:",total_game_num)

if __name__ == "__main__":
    main()




