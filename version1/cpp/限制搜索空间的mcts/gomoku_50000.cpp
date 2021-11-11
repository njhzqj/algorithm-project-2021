/*

	the plan of re-write MCTS in cpp to get better results

	start: 2021/10/30

	notes:

	------------

	2021/10/30: write some helper functions, define two structs, prepare to write mctsSearch

	2021/11/02: write the functions for MCTS, consider the memory problems

	2021/11/05: consider to restrict the search space to get better performance

*/

#include <iostream>

#include <cstdlib>

#include <vector>

#include <time.h>

#include <math.h>

#include <cstring>

#include <sstream>

#include <algorithm>

#include "jsoncpp/json.h"


using namespace std;

struct Action {

	int x;

	int y;

	char player;

};

struct Node {

	char** grid;

	struct Node* parent;

	vector<Node*> child;

	int child_number;

	int n;

	int q;

	struct Action* act;

};

const int WIN = 5;

const int SIZE = 15;

const char SPACE = ' ';

const char WHITE = 'W';

const char BLACK = 'B';

const int DISTANCE = 1;

const int RESTRICT = 35000;

char** grid;

Node* start_state;

//bool isAvailablePosition(int x,int y, char** c_grid) {
//	int min_distance = SIZE + SIZE - 2;
//	for (int i = 0; i < SIZE; i++) {
//		for (int j = 0; j < SIZE; j++) {
//			if (c_grid[i][j] != SPACE) {
//				int c_distance = abs(x - i) + abs(y - j);
//				if (c_distance < min_distance) {
//					min_distance = c_distance;
//				}
//			}
//		}
//	}
//	if (min_distance < DISTANCE) {
//		return true;
//	}
//	else {
//		return false;
//	}
//}



char getOponent(char player) {

	if (player == BLACK) {

		return WHITE;

	}

	else if (player == WHITE) {

		return BLACK;

	}

	else {

		return SPACE;

	}

}

// allocate memory and init the root node

Node* initNewRootNode(char player) {

	Node* state = new Node();

	state->parent = NULL;

	state->child_number = 0;

	Action* act = new Action();

	act->x = -1;

	act->y = -1;

	act->player = getOponent(player);

	state->act = act;

	state->grid = (char**)malloc(sizeof(char*) * SIZE);

	for (int i = 0; i < SIZE; i++) {

		state->grid[i] = (char*)malloc(sizeof(char) * SIZE);

	}

	for (int i = 0; i < SIZE; i++) {

		for (int j = 0; j < SIZE; j++) {

			state->grid[i][j] = grid[i][j];

		}

	}

	return state;

}

// allocate memory and init a new node based on the given node and action

Node* newNodeWithParent(Node* parent, Action* action) {

	Node* state = new Node();

	state->parent = parent;

	state->child_number = 0;

	state->act = action;

	state->grid = (char**)malloc(sizeof(char*) * SIZE);

	for (int i = 0; i < SIZE; i++) {

		state->grid[i] = (char*)malloc(sizeof(char) * SIZE);

	}

	for (int i = 0; i < SIZE; i++) {

		for (int j = 0; j < SIZE; j++) {

			state->grid[i][j] = parent->grid[i][j];

			if (action->x == i && action->y == j) {

				state->grid[i][j] = action->player;

			}

		}

	}

	parent->child_number += 1;

	parent->child.push_back(state);

	return state;

}

// free node's grid and all its child nodes

void freeNode(Node* state) {

	if (state->child_number > 0) {

		for (int i = 0; i < state->child_number; i++) {

			freeNode(state->child[i]);

		}

	}

	for (int i = 0; i < SIZE; i++) {

		free(state->grid[i]);

	}

	free(state->grid);

	delete state->act;

	delete state;

}

// allocate memory space for grid

void initGrid() {

	grid = (char**)malloc(sizeof(char*) * SIZE);

	for (int i = 0; i < SIZE; i++) {

		grid[i] = (char*)malloc(sizeof(char) * SIZE);

	}

	for (int i = 0; i < SIZE; i++) {

		for (int j = 0; j < SIZE; j++) {

			grid[i][j] = SPACE;

		}

	}

}

// free grid

void freeGrid() {

	for (int i = 0; i < SIZE; i++) {

		free(grid[i]);

	}

	free(grid);

}

// this function is useless in this code version
//int getAvailActNums(char** c_grid) {
//
//	int count = 0;
//
//	for (int i = 0; i < SIZE; i++) {
//
//		for (int j = 0; j < SIZE; j++) {
//
//			if (c_grid[i][j] == SPACE) {
//
//				count++;
//
//			}
//
//		}
//
//	}
//
//	return count;
//
//}

vector<pair<int, int>> getAvailActs(char** c_grid) {
	vector<pair<int, int>> acts;
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			if (c_grid[i][j] == SPACE) {
				int l = i - 1;
				int r = i + 1;
				int u = j - 1;
				int d = j + 1;
				if (l < 0) {
					l = 0;
				}
				if (r >= SIZE) {
					r = SIZE - 1;
				}
				if (u < 0) {
					u = 0;
				}
				if (d >= SIZE) {
					d = SIZE - 1;
				}
				if (c_grid[l][u] != SPACE || c_grid[i][u] != SPACE || c_grid[r][u] != SPACE || c_grid[l][j] != SPACE || c_grid[r][j] != SPACE || c_grid[l][d] != SPACE || c_grid[i][d] != SPACE || c_grid[r][d] != SPACE) {
					acts.push_back(make_pair(i,j));
				}
			}
		}
	}
	random_shuffle(acts.begin(), acts.end());
	/*cout << "## acts:" << endl;
	for (int i = 0; i < acts.size(); i++) {
		cout << acts[i].first << ',' << acts[i].second << endl;
	}*/
	return acts;
}

void printGrid(char** c_grid) {

	cout << endl;

	for (int i = 0; i < SIZE; i++) {

		if (i == 0) {

			for (int j = 0; j < SIZE; j++) {

				cout << "-";

			}

			cout << endl;

		}

		for (int j = 0; j < SIZE; j++) {

			cout << c_grid[i][j];

		}

		cout << endl;

		if (i == SIZE - 1) {

			for (int j = 0; j < SIZE; j++) {

				cout << "-";

			}

			cout << endl;

		}

	}

	cout << endl;

}

bool isGridFull(char** c_grid) {

	for (int i = 0; i < SIZE; i++) {

		for (int j = 0; j < SIZE; j++) {

			if (c_grid[i][j] == SPACE) {

				return false;

			}

		}

	}

	return true;

}

bool isGridEmpty() {

	int count = 0;

	for (int i = 0; i < SIZE; i++) {

		for (int j = 0; j < SIZE; j++) {

			if (grid[i][j] == SPACE) {

				count++;

			}

		}

	}

	if (count == SIZE * SIZE) {

		return true;

	}

	else {

		return false;

	}

}

bool inGrid(int x, int y) {

	if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {

		return false;

	}

	return true;

}

int manDistance(int x1, int y1, int x2, int y2) {

	return abs(x1 - x2) + abs(y1 - y2);

}

int checkTowards(char** c_grid, int xx, int yy, int x, int y, char player) {

	int i = 1;

	int j = 1;

	if (!inGrid(x, y)) {

		return 0;

	}

	for (; i <= WIN; i++) {

		int temp_x = x + xx * i;

		int temp_y = y + yy * i;

		if (!inGrid(temp_x, temp_y) || c_grid[temp_x][temp_y] != player) {

			break;

		}

	}

	for (; j <= WIN; j++) {

		int temp_x = x - xx * j;

		int temp_y = y - yy * j;

		if (!inGrid(temp_x, temp_y) || c_grid[temp_x][temp_y] != player) {

			break;

		}

	}

	return i + j - 1;

}

bool isWin(char** c_grid, Action* act) {

	return checkTowards(c_grid, 1, 0, act->x, act->y, act->player) >= 5 || checkTowards(c_grid, 0, 1, act->x, act->y, act->player) >= 5 || checkTowards(c_grid, 1, 1, act->x, act->y, act->player) >= 5 || checkTowards(c_grid, 1, -1, act->x, act->y, act->player) >= 5;

}

bool placePiece(Action* act) {

	if (act->player != BLACK && act->player != WHITE) {

		return false;

	}

	if (grid[act->x][act->y] != SPACE) {

		return false;

	}

	grid[act->x][act->y] = act->player;

	return true;

}

void printTree(Node* state) {

	cout << "state info:" << endl;

	cout << "  child: " << state->child_number << " q: " << state->q << " n: " << state->n << endl;

	//for (int i = 0; i < state->child_number; i++) {

	//	cout << "  " << i << "th child, " << state->child[i]->act->x << "," << state->child[i]->act->y << state->child[i]->act->player << endl;

	//	cout << "     its  child nums: " << state->child[i]->child_number << " q: " << state->child[i]->q << " n: " << state->child[i]->n << endl;

	//	// printGrid(state->child[i]->grid);

	//}

}

// functions for monte carlo tree search

// this function is useless in the code version
//Node* expand(Node* state) {
//
//	vector<pair<int, int>> availAct = getAvailActs(state->grid);
//
//	int n = availAct.size();
//
//	//cout << "act size:" << n << endl;
//
//	Node* new_state;
//
//	
//
//	int x, y;
//
//	x = availAct[0].first;
//
//	y = availAct[0].second;
//
//	Action* act = new Action();
//
//	act->x = x;
//
//	act->y = y;
//
//	act->player = getOponent(state->act->player);
//
//	new_state = newNodeWithParent(state, act);
//	
//	
//
//	return new_state;
//}

Node* bestChild(Node* state, double c) {

	double max_score;

	int max_index;

	for (int i = 0; i < state->child_number; i++) {

		double s;

		if (c != 0) {

			s = double(state->child[i]->q) / double(state->child[i]->n) + c * sqrt(log(double(state->n)) * 2 / double(state->child[i]->n));

		}
		else {
			s = double(state->child[i]->q) / double(state->child[i]->n);
		}

		if (isWin(state->child[i]->grid, state->child[i]->act) && c == 0) {

			max_index = i;

			break;

		}

		else if (i == 0) {

			max_index = i;

			max_score = s;

		}

		else if (s > max_score) {

			max_index = i;

			max_score = s;

		}

	}

	return state->child[max_index];

}

Node* newNode(Node* state, double c) {

	Node* c_state = state;


	while (!isGridFull(c_state->grid) && !isWin(c_state->grid, c_state->act)) {

		vector<pair<int, int>> availAct = getAvailActs(c_state->grid);

		int n = availAct.size();

		// cout << " act size:" << n << endl;

		if (n > c_state->child_number) {

			//return expand(c_state);

			// the expand() step

			Node* new_state;

			int x, y;

			x = availAct[0].first;

			y = availAct[0].second;

			Action* act = new Action();

			act->x = x;

			act->y = y;

			act->player = getOponent(c_state->act->player);

			new_state = newNodeWithParent(c_state, act);

			return new_state;
		}

		else {

			c_state = bestChild(c_state, c);

		}

	}

	return c_state;

}

int simulation(Node* state) {

	char** temp_grid;

	temp_grid = (char**)malloc(sizeof(char*) * SIZE);

	for (int i = 0; i < SIZE; i++) {

		temp_grid[i] = (char*)malloc(sizeof(char) * SIZE);

	}

	for (int i = 0; i < SIZE; i++) {

		for (int j = 0; j < SIZE; j++) {

			temp_grid[i][j] = state->grid[i][j];

		}

	}

	Action* act = new Action();

	act->player = state->act->player;

	while (!isGridFull(temp_grid)) {

		act->x = rand() % (SIZE);

		act->y = rand() % (SIZE);

		if (temp_grid[act->x][act->y] == SPACE) {

			// printGrid(temp_grid);

			act->player = getOponent(act->player);

			temp_grid[act->x][act->y] = act->player;

			if (isWin(temp_grid, act)) {

				if (act->player == start_state->act->player) {

					for (int i = 0; i < SIZE; i++) {

						free(temp_grid[i]);

					}

					free(temp_grid);

					delete act;

					return 0;

				}

				else {

					for (int i = 0; i < SIZE; i++) {

						free(temp_grid[i]);

					}

					free(temp_grid);

					delete act;

					return 2;

				}

			}

		}

	}

	for (int i = 0; i < SIZE; i++) {

		free(temp_grid[i]);

	}

	free(temp_grid);

	delete act;

	return 1;

}

void backup(Node* state, int reward) {

	Node* c_state = state;

	c_state->n += 2;

	c_state->q += reward;

	while (c_state->parent != NULL) {

		c_state->parent->n += 2;

		c_state->parent->q += reward;

		c_state = c_state->parent;

	}

}

Action* mctsSearch(int restrict, char player) {

	double c = 1 / sqrt(2);

	Action* act = new Action();

	if (isGridEmpty()) {

		act->player = player;

		act->x = int(SIZE / 2);

		act->y = int(SIZE / 2);

		return act;

	}

	else {

		//Node* start_state;

		// clock_t start, end1, end2, end3;

		// int new_node_time = 0;
		
		// int simulation_time = 0;

		// int backup_time = 0;

		start_state = initNewRootNode(player);

		for (int i = 0; i < restrict; i++) {

			// start = clock();

			Node* new_state = newNode(start_state, c);

			// end1 = clock();

			// new_node_time += (end1 - start);

			int reward = simulation(new_state);

			// end2 = clock();

			// simulation_time += end2 - end1;

			backup(new_state, reward);

			// end3 = clock();

			// backup_time += end3 - end2;

		}

		// cout << new_node_time << endl;
		// cout << simulation_time << endl;
		// cout << backup_time << endl;

		Node* best_child = bestChild(start_state, 0.0);

		act->player = best_child->act->player;

		act->x = best_child->act->x;

		act->y = best_child->act->y;

		// printTree(start_state);

		freeNode(start_state);

		return act;

	}

}

void offline_testing() {

	clock_t start, end;

	// set srand()

	srand((unsigned)time(NULL));

	// init grid

	initGrid();

	Action* mcts_act;

	printGrid(grid);

	char player = WHITE;

	int count = 0;

	//while(count<2){
	while (true) {

		start = clock();

		player = getOponent(player);

		cout << player << " player's round: " << count << endl;

		mcts_act = mctsSearch(RESTRICT, player);

		cout << mcts_act->x << "," << mcts_act->y << "," << mcts_act->player << endl;

		placePiece(mcts_act);

		printGrid(grid);

		if (isWin(grid, mcts_act)) {

			cout << mcts_act->player << " win the game!" << endl;

			break;

		}

		delete mcts_act;

		count++;

		end = clock();

		cout << "decision time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;  //输出时间（单位：ｓ）

	}

	// free grid

	freeGrid();

}

void onlineGaming() {

	// init grid

	initGrid();

	// pointer to action
	Action* mcts_act;

	// player 
	char player;

	string str;
	getline(cin, str);
	Json::Value input;
	Json::Reader().parse(str, input);
	const int turnID = input["requests"].size();
	const bool isBlack = input["requests"][0u]["x"].asInt() == -1;

	if (isBlack) {
		player = BLACK;
	}
	else {
		player = WHITE;
	}

	for (int i = 0; i < turnID; i++)
	{
		if (!isBlack || i > 0)
		{
			int x1 = input["requests"][i]["x"].asInt();
			int y1 = input["requests"][i]["y"].asInt();
			grid[x1][y1] = getOponent(player);
		}
		if (i == turnID - 1)
			break;

		int x2 = input["responses"][i]["x"].asInt();
		int y2 = input["responses"][i]["y"].asInt();
		grid[x2][y2] = player;
	}

	Json::Value output;


	mcts_act = mctsSearch(RESTRICT, player);

	output["response"]["x"] = mcts_act->x;
	output["response"]["y"] = mcts_act->y;

	delete mcts_act;

	cout << Json::FastWriter().write(output) << endl;


	// free grid

	freeGrid();


}


int main() {

	// offline_testing();


	onlineGaming();

	return 0;

}