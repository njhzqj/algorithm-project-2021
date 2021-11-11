import random
import math
import copy

SIZE = 15
EMPTY = " "
WHITE = "W"
BLACK = "B"
NULL = "NULL"

def ma_distance(x1,y1,x2,y2):
    """
    return the manhatan distance between (x1,y1) and (x2,y2)
    """
    return abs(x1-x2) + abs (y1-y2)

class Grid:
    """
    this class is to repr a gird of a gomoku game
    self.size: the size of the grid
    self.grid: a 2-d list to store the grid info
                EMPTY means there is no piece in the given position
                WHITE means there is a white piece in the given position
                BLACK means there is a black piece in the given position
    """
    def __init__(self,size):
        """
        init the grid with a given size
        """
        self.size = size
        self.grid = [[EMPTY for i in range(size)] for j in range(size)]

    def inGrid(self,x,y):
        """
        check if a given x,y is out from the grid
        """
        if x < 0 or x >= self.size:
            return False
        if y < 0 or y >= self.size:
            return False
        return True

    def isFull(self):
        """
        check if this grid is full of pieces
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.isEmpty(i,j):
                    return False
        return True
    
    def isAllEmpty(self):
        """
        check if there is no piece in grid
        """
        for i in range(self.size):
            for j in range(self.size):
                if not self.isEmpty(i,j):
                    return False
        return True

    def isEmpty(self,x,y):
        """
        check if there is a piece in the given position (x,y)
        """
        if not self.inGrid(x,y):
            return False
            #raise ValueError("(x,y) should be in range of the grid!")
        if self.grid[x][y] == EMPTY:
            return True
        else:
            return False

    def place(self,x,y,piece):
        """
        place a piece to the given position in the gird
        if there is already a piece in the given place, if will return False
        else thie function will place the piece into the grid and return True
        """
        if not self.inGrid(x,y):
            return False
            #raise ValueError("(x,y) should be in range of the grid!")
        if piece not in [BLACK,WHITE]:
            raise ValueError("a piece shall be BLACK or WHITE!")
        if not self.isEmpty(x,y):
            return False
        self.grid[x][y] = piece
        return True

    def remove(self,x,y):
        """
        remove a piece from the given position in the gird
        if there is no piece in the given place, if will return False
        else thie function will remove the piece from the grid and return True
        """
        if not self.inGrid(x,y):
            return False
            #raise ValueError("(x,y) should be in range of the grid!")
        if self.isEmpty(x,y):
            return False
        self.grid[x][y] = EMPTY
        return True

    def get(self,x,y):
        """
        return the piece in the given position (x,y) in the grid
        """
        if not self.inGrid(x,y):
            raise ValueError("(x,y) should be in range of the grid!")
        return self.grid[x][y]

    def getAvailablePositions(self):
        """
        """
        availList = []
        for i in range(self.size):
            for j in range(self.size):
                if self.isEmpty(i,j):
                    availList.append((i,j))
        return availList

    def maDistanceToNearestPiece(self,x,y):
        """
        return the ma distance from (x,y) to the nearest piece in the grid
        """
        minDis = 2 * (self.size-1)
        for i in range(self.size):
            for j in range(self.size):
                if not self.isEmpty(i,j):
                    manDis = ma_distance(x,y,i,j)
                    if manDis<minDis:
                        minDis = manDis
        return minDis

# is win function problem!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!
    def isWinAfterPlace(self,x,y,player):
        """
        """
        def checkTowards(xx,yy):
            for i in range(6):
                temp_x = x + xx * i
                temp_y = y + yy * i
                if (not self.inGrid(temp_x,temp_y)) or self.get(temp_x,temp_y)!=player:
                    if i != 0:
                        break
                #print(temp_x,temp_y,self.get(temp_x,temp_y))
            for j in range(6):
                temp_x = x - xx * j
                temp_y = y - yy * j
                if (not self.inGrid(temp_x,temp_y)) or self.get(temp_x,temp_y)!=player:
                    if j != 0:
                        break
                #print(temp_x,temp_y,self.get(temp_x,temp_y))
            #print("num",i,j,i+j-1)
            return i + j -1
        #print(checkTowards(1,0),checkTowards(0,1),checkTowards(1,1),checkTowards(1,-1))
        return checkTowards(1,0) >= 5 or checkTowards(0,1) >= 5 or checkTowards(1,1) >= 5 or checkTowards(1,-1) >= 5
        
    def isWin(self):
        """
        check the current grid to justify if some player wins this game
        return EMPTY if no one wins
        return WHITE if white player wins
        return BLACK if black player wins
        """
        whiteWin = "".join([WHITE for i in range(5)])
        blackWin = "".join([BLACK for i in range(5)])

        # check every col
        for i in range(self.size):
            if ("".join([self.grid[i][j] for j in range(self.size)])).find(whiteWin) != -1:
                return WHITE
            if ("".join([self.grid[i][j]for j in range(self.size)])).find(blackWin) != -1:
                return BLACK

        # check every row
        for i in range(self.size):
            if ("".join([self.grid[j][i] for j in range(self.size)])).find(whiteWin) != -1:
                return WHITE
            if ("".join([self.grid[j][i] for j in range(self.size)])).find(blackWin) != -1:
                return BLACK

        # check diagonls
        fdiag = [[] for _ in range(2 * self.size - 1)]
        bdiag = [[] for _ in range(2 * self.size - 1)]
        min_bdiag = 1 - self.size

        for x in range(self.size):
            for y in range(self.size):
                fdiag[x+y].append(self.grid[y][x])
                bdiag[x-y-min_bdiag].append(self.grid[y][x])
        
        # for i in range(self.size):
        for i in range(2 * self.size - 1):
            if ("".join(fdiag[i])).find(whiteWin) != -1:
                return WHITE
            if ("".join(bdiag[i])).find(whiteWin) != -1:
                return WHITE
            if ("".join(fdiag[i])).find(blackWin) != -1:
                return BLACK
            if ("".join(bdiag[i])).find(blackWin) != -1:
                return BLACK
            
        return EMPTY

    def str_repr(self):
        """
        store current game info into a str for command line viewing 
        """
        repr = "\n##CURRENT##GRID##\n"
        repr += "#"
        for i in range(self.size):
            repr += str(i%10)
        repr += "#"
        repr += "\n"
        for i in range(self.size):
            repr += str(i%10)
            for j in range(self.size):
                repr += self.grid[i][j]
            repr += "#\n"
        for i in range(self.size+2):
            repr += "#"
        repr += "\n"
        return repr

    def getNearbyAvailActs(self):
        """
        get all nearby avail acts
        """
        if self.isAllEmpty():
            return [(int(self.size/2),int(self.size/2))]
        nearbyAvailActs = []
        for i in range(self.size):
            for j in range(self.size):
                if self.isEmpty(i,j) and self.maDistanceToNearestPiece(i,j) <= 2:
                    nearbyAvailActs.append((i,j))
        random.shuffle(nearbyAvailActs)
        return nearbyAvailActs

class GomokuBase:
    """
    this class is the base class for my gomoku game solver
    """
    def __init__(self,size):
        """
        init a game with a given size
        """
        self.size = size
        self.grid = Grid(self.size)

    def act(self,x,y,piece):
        """
        """
        return self.grid.place(x,y,piece)

    def randomAct(self):
        """
        randomly return one action from all available places
        """
        availActs = self.getNearbyAvailActs()
        act = availActs[random.randint(0,len(availActs)-1)]
        return act


    def getNearbyAvailActs(self):
        """
        get all nearby avail acts
        """
        nearbyAvailActs = []
        allAvailActs = self.grid.getAvailablePositions()
        if self.grid.isAllEmpty():
            return [(int(self.size/2),int(self.size/2))]
        else:
            for act in allAvailActs:
                if self.grid.maDistanceToNearestPiece(act[0],act[1]) < 3:
                    nearbyAvailActs.append(act)
        random.shuffle(nearbyAvailActs)
        return nearbyAvailActs

class GomokuState:
    """
    this class is used to store a state of GomokuState
    """
    def __init__(self,parent,action,grid=NULL,player=NULL):
        """
        """
        if player != NULL:
            if player == WHITE:
                self.player = BLACK
            else:
                self.player = WHITE
        else:
            if parent.player == WHITE:
                self.player = BLACK
            else:
                self.player = WHITE
        self.action = action
        self.parent = parent
        if grid == NULL:
            self.grid = copy.deepcopy(parent.grid)
        else:
            self.grid = grid
        if action!= NULL:
            self.grid.place(*(action),self.player)
        self.childs = []
        self.qu = 0.0
        self.num = 0.0

    def add_child(self,child):
        """
        """
        self.childs.append(child)


class MCTS_Gomoku(GomokuBase):
    """
    """
    def showTree(self,state):
        """
        """
        print("  Search: ",int(state.qu/2),"wins in",int(state.num/2),"simulations.","child numbers:",len(state.childs))
        for child in state.childs:
            print("  child qu:{} num:{} player:{} action:{}".format(child.qu,child.num,child.player,child.action))


    def search(self,c,search_times,player):
        """
        """
        if self.grid.isAllEmpty():
            return (int(self.size/2),int(self.size/2))
        else:
            self.player = player
            start_state = GomokuState(NULL,NULL,grid=self.grid,player=self.player)
            self.availActs = start_state.grid.getNearbyAvailActs()
            for i in range(search_times):
                new_state = self.newNode(start_state,c)
                x,y = new_state.action[0],new_state.action[1]

                if new_state.grid.isWinAfterPlace(x,y,self.player):
                    return new_state.action
                else:
                    reward = self.simulation(new_state)

                start_state = self.backup(new_state,reward)


            self.showTree(start_state)
            return self.bestChild(start_state,c).action

    def newNode(self,state,c):
        """
        """
        
        temp_state = copy.deepcopy(state)
        x,y = 7,7
        p = temp_state.player
        # import time
        #s = time.time()
        while not temp_state.grid.isFull() and not temp_state.grid.isWinAfterPlace(x,y,p):
            if len(temp_state.grid.getNearbyAvailActs()) > len(temp_state.childs):
                #print(time.time()-s)
                return self.expand(temp_state)
            else:
                temp_state = self.bestChild(temp_state,c)
                x,y = temp_state.action[0],temp_state.action[1]
                p = temp_state.player
        #print(time.time()-s)
        return temp_state

        #####
        if len(self.availActs) > len(state.childs):
            return self.expand(state)
        else:
            return self.bestChild(state,c)
        
            

    def bestChild(self,state,c):
        """
        """
        for index,child in enumerate(state.childs):
            temp = math.log(state.num,math.e) * 2 / child.num
            score = child.qu / child.num  + c * (temp)**0.5
            if index == 0:
                max_score = score
                max_child = child
            else:
                if score > max_score:
                    max_score = score
                    max_child = child
        return max_child

    def expand(self,state):
        """
        """
        availActs = self.availActs
        exist_acts = [child.action for child in state.childs]
        for i in range(len(availActs)):
            act = availActs[i]
            if act not in exist_acts:
                break
        new_state = GomokuState(state,act)
        state.add_child(new_state)
        return new_state

    def simulation(self,state):
        """
        """
        temp_grid = copy.deepcopy(state.grid)
        p = state.player
        while not temp_grid.isFull():
            x = random.randint(0,self.size-1)
            y = random.randint(0,self.size-1)
            if temp_grid.isEmpty(x,y):
                if p == WHITE:
                    p = BLACK
                else:
                    p = WHITE
                if temp_grid.isWinAfterPlace(x,y,p):
                    if p == self.player:
                        return 2
                    else:
                        return 0
                temp_grid.place(x,y,p)
        return 1

    def backup(self,state,reward):
        """
        """
        state.num += 2
        state.qu += reward
        temp_state = copy.deepcopy(state)
        while temp_state.parent != NULL:
            temp_state.parent.num += 2
            temp_state.parent.qu += reward
            temp_state = temp_state.parent
        return temp_state

def jsonToAct(jsonText):
    return (int(jsonText["x"]),int(jsonText["y"]))

def onlineGame():
    import json
    game = MCTS_Gomoku(15)
    full_input = json.loads(input())
    if "data" in full_input:
        my_data = full_input["data"]
    else:
        my_data = None

    all_requests = full_input["requests"]
    all_responses = full_input["responses"]
    
    for i in range(len(all_responses)):
        myInput = all_requests[i]
        myOutput = all_responses[i]
        game.act(*jsonToAct(myInput),BLACK)
        game.act(*jsonToAct(myOutput),WHITE)

    curr_input = all_requests[-1]
    game.act(*jsonToAct(curr_input),BLACK)

    c = 1 / math.sqrt(2.0)
    act = game.search(c,100,WHITE)

    my_action = {"x":act[0],"y":act[1]}

    print(json.dumps({
        "response": my_action,
        "data": my_data
    }))


if __name__ == "__main__":
    onlineGame()
