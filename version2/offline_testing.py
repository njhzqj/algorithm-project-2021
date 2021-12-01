"""
this file is for testing the models and AIs on the local computer
"""
from mcts import *


def main():
    vp = ValuePolicy()
    vp.load_model("save/saved_models/high_abc_20000/abc_policy_value_e_1.pt")
    mctsSeacher = MctsSeacher(vp)
    grid = [[SPACE for _ in range(GRIDSIZE)] for _ in range(GRIDSIZE)]
    
    
    player = WHITE_PLAYER
    count = 0

    postacts = []

    while count < 15:
        act = mctsSeacher.mctsSearch(grid,player)
        player = getOponent(player)
        if act in postacts:
            print("\n!! invalid act!!\n")
        print("act place",act)
        postacts.append(act)
        grid[act[0]][act[1]] = player
        print(gridToStr(grid))
        count += 1

if __name__ == "__main__":
    main()


# 现在的两个问题： 1. Invalid Move 2. 训练集镜像和反转