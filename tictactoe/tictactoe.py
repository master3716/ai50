"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0

    for i in board:
        for j in i:
            if(j == X):
                x_count += 1
            elif(j == O):
                o_count += 1

    return X if x_count - o_count == 0 else O
            


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possibilities = set()

    for i in range(len(board)):
        for j in range(len(board[0])):
            if(board[i][j] == EMPTY):
                possibilities.add((i, j))

    return possibilities



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    possible_actions = actions(board)

    if action not in possible_actions:
        raise Exception("Not a possible move")
    
    turn_player = player(board)

    new_board = copy.deepcopy(board)

    if turn_player == X:
        new_board[action[0]][action[1]] = X
    else:
        new_board[action[0]][action[1]] = O

    return new_board





def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    for j in range(len(board)):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] is not EMPTY:
            return board[0][j]
        
    for i in range(len(board)):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            return board[i][0]
        
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]
    
    return None

    

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if(winner(board)):
        return True
    
    for i in board:
        if EMPTY in i:
            return False
    return True
    


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    res = winner(board)
    if(not res):
        return 0
    elif(res == X):
        return 1
    else:
        return -1


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if(terminal(board)):
        return None
    turn_player = player(board)
    winning_action = ()

    if turn_player == X:
        max_v = -math.inf
        for action in actions(board):
            res = result(board, action)
            val = Min_value(res)
            if(val > max_v):
                winning_action = action
                max_v = val

    else:
        min_v = math.inf
        for action in actions(board):
            res = result(board, action)
            val = Max_Value(res)
            if(val < min_v):
                winning_action = action
                min_v = val

    return winning_action
        

def Max_Value(board):
    if(terminal(board)):
        return utility(board)
    
    max_v = -math.inf
    for action in actions(board):
        res = result(board, action)
        max_v = max(Min_value(res), max_v)
    return max_v

def Min_value(board):
    if(terminal(board)):
        return utility(board)
    
    min_v = math.inf
    for action in actions(board):
        res = result(board, action)
        min_v = min(Max_Value(res), min_v)
    return min_v
