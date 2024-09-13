#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:28:49 2020

@author: navid
"""

import numpy as np
import random
import os


UP = np.array((0,1))
DOWN = np.array((0,-1))
LEFT = np.array((-1,0))
RIGHT = np.array((1,0))

UP_RIGHT = np.array((1,1))
DOWN_RIGHT = np.array((1,-1))
UP_LEFT = np.array((-1,1))
DOWN_LEFT = np.array((-1,-1))

DIRECTIONS = [UP, DOWN, LEFT, RIGHT, UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT]


#%%
#creates the iniital board
def creat_board(size = 8):
    board = np.zeros((size,size))
    middle = int(size/2)
    board[middle-1][middle-1],board[middle][middle] = -1,-1
    board[middle-1][middle],board[middle][middle-1] = 1,1
    return board
#%%
#creates a matrix showing the possible moves at each state for player i
#also the number of pieces that the corresponding move will change (immidiate reward)
def avail_matrix(size):
    avail = np.zeros((size,size))
    return avail


def check_direction(board, player, position, direction, rewards):
    size = board.shape[0]
    new_pos = tuple(position + direction)
    
    if (new_pos[0] >= 0 and new_pos[0] < size and
        new_pos[1] >= 0 and new_pos[1] < size):
        if board[new_pos] == -player:
            rewards += 1
            return check_direction(board, player, new_pos, direction, rewards)
        elif board[new_pos] == player:
            return False , 0 , position
        elif board[new_pos] == 0:
            if rewards != 0:
                return True , rewards , new_pos
            else:
                return False , 0 , position
    else:
        return False , 0 , position


    
    
def get_turn_list_from_direction(board , player , position, direction, turning_list, counter):
    size = board.shape[0]
    new_pos = tuple(position + direction)
    
    if (new_pos[0] >= 0 and new_pos[0] < size and
        new_pos[1] >= 0 and new_pos[1] < size):
        if board[new_pos] == -player:
            counter+=1
            turning_list.append(new_pos)
            return get_turn_list_from_direction(board , player , new_pos, direction, turning_list, counter)
        elif board[new_pos] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop() # Remove last
            return turning_list
    else:
        for i in range(counter):
            turning_list.pop()
        return turning_list
    

        
#updates the v_matrix  
def valid_moves(board , player):
    avail = avail_matrix(board.shape[0])
    result = np.where(board==player)
    for coordinate in zip(result[0],result[1]):
        for d in DIRECTIONS:
            possibility , rewards , possible_coordinate = check_direction(board, player, coordinate, d, 0)
            if possibility:
                avail[possible_coordinate] += rewards
    return avail

def get_turn_list(move , board , player):
    turn_list = []
    for d in DIRECTIONS:
        turn_list = get_turn_list_from_direction(board , player , move, d, turn_list, 0)
    return turn_list




if __name__ == "__main__":
    random.seed(50)

    board = creat_board(8)
    print(board)
    os.system('clear')
    print(board)
    player = -1
    while True:    
        """
        player = -1
        avail = valid_moves(board,player)

        if np.amax(avail) == 0:
            print('No more move is available!')
            break
        else:        
            dummy=np.where(avail==np.amax(avail))
            maxavail = list(zip(dummy[0],dummy[1]))
            x_move = int(input('Enter the row: '))
            while True:
                if x_move in dummy[0]:
                    break
                else:
                    print('This is not a valid move! Try another')
                    x_move = int(input('Enter the row: '))        

            y_move = int(input('Enter the column: '))
            while True:
                if y_move in dummy[1]:
                    break
                else:
                    print('This is not a valid move! Try another')
                    y_move = int(input('Enter the column: '))  
            move = (x_move,y_move)
            turn_list = get_turn_list(move, board , player)
            board[move] = player
            for i in turn_list:
                board[i] = player
        """

        # 
        player *= -1
        avail =  valid_moves(board , player)
        if np.amax(avail) == 0:
            break
        else:
            dummy=np.where(avail==np.amax(avail))
            maxavail = list(zip(dummy[0],dummy[1]))
            move = random.choice(maxavail)
            turn_list = get_turn_list(move, board , player)
            board[move] = player
            for i in turn_list:
                board[i] = player

        print(board)


    unique, counts = np.unique(board, return_counts=True)
    evaluation = dict(zip(unique, counts))
    print(evaluation)