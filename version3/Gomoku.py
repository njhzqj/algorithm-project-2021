"""
The file contains some useful functions and varaiabels of the gomoku game.
"""
import numpy as np

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