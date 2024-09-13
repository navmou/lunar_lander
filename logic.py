#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:28:49 2020

@author: navid
"""

import numpy as np
import random
import os
#%%
#creates the iniital board
def creat_board(size):
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

#%%checking available move and immidiate reward for each available position
    
#check the direction vertical minus (down) on the ordinary viewed board
def vertical_minus(board , player , x , y , rewards):
    if x < 7:    
        if board[x+1][y] == -player:
            rewards+=1
            return vertical_minus(board,player,x+1,y,rewards)
        elif board[x+1][y] == player:
            return False , 0 , (x,y)
        elif board[x+1][y] == 0:
            if rewards != 0:
                return True , rewards , (x+1,y)
            else:
                return False , 0 , (x,y)
    else:
        return False , 0 , (x,y)
            
#check the direction vertical plus (up) on the ordinary viewed board
def vertical_plus(board , player , x , y , rewards):
    if x > 0:    
        if board[x-1][y] == -player:
            rewards+=1
            return vertical_plus(board,player,x-1,y,rewards)
        elif board[x-1][y] == player:
            return False , 0 , (x,y)
        elif board[x-1][y] == 0:
            if rewards != 0:
                return True , rewards , (x-1 , y)
            else:
                return False , 0 , (x,y)
    else:
        return False , 0 , (x,y)
#check the direction horizontal plus (right) on the ordinary viewed board
def horizontal_plus(board , player , x , y , rewards):
    if y < 7:
        if board[x][y+1] == -player:
            rewards+=1
            return horizontal_plus(board,player,x,y+1,rewards)
        elif board[x][y+1] == player:
            return False , 0 , (x,y)
        elif board[x][y+1] == 0:
            if rewards == 0:
                return False , 0 , (x,y)
            else:
                return True , rewards , (x, y+1)
    else:
        return False , 0 , (x,y)
#check the direction horizontal minus (left) on the ordinary viewed board
def horizontal_minus(board , player , x , y , rewards):
    if y > 0:
        if board[x][y-1] == -player:
            rewards+=1
            return horizontal_minus(board,player,x,y-1,rewards)
        elif board[x][y-1] == player:
            return False , 0 , (x,y)
        elif board[x][y-1] == 0:
            if rewards == 0:
                return False , 0 , (x,y)
            else:
                return True,rewards , (x,y-1)
    else:
        return False , 0 , (x,y)
        
#check the direction diagonal x=y to left on the ordinary viewed board            
def diagonal_minus_1(board , player , x , y , rewards):
    if x < 7 and y > 0:
        if board[x+1][y-1] == -player:
            rewards+=1
            return diagonal_minus_1(board,player,x+1,y-1,rewards)
        elif board[x+1][y-1] == player:
            return False , 0 , (x,y)
        elif board[x+1][y-1] == 0:
            if rewards == 0:
                return False , 0 , (x,y)
            else:
                return True,rewards, (x+1 , y-1)
    else:
        return False , 0 , (x,y)
#check the direction diagonal x=y to right on the ordinary viewed board                        
def diagonal_plus_1(board , player , x , y , rewards):
    if x > 0  and y < 7:
        if board[x-1][y+1] == -player:
            rewards+=1
            return diagonal_plus_1(board,player,x-1,y+1,rewards)
        elif board[x-1][y+1] == player:
            return False , 0 , (x,y)
        elif board[x-1][y+1] == 0:
            if rewards == 0:
                return False , 0 , (x,y)
            else:
                return True,rewards, (x-1,y+1)
    else:
        return False , 0 , (x,y)
#check the direction diagonal x=-y to left on the ordinary viewed board            
def diagonal_plus_2(board , player , x , y , rewards):
    if x > 0 and y > 0:
        if board[x-1][y-1] == -player:
            rewards+=1
            return diagonal_plus_2(board,player,x-1,y-1,rewards)
        elif board[x-1][y-1] == player:
            return False , 0 , (x,y)
        elif board[x-1][y-1] == 0:
            if rewards == 0:
                return False , 0 , (x,y)
            else:
                return True,rewards , (x-1,y-1)
    else:
        return False , 0 , (x,y)
#check the direction diagonal x=-y to right on the ordinary viewed board            
def diagonal_minus_2(board , player , x , y , rewards):
    if x < 7 and y < 7:
        if board[x+1][y+1] == -player:
            rewards+=1
            return diagonal_minus_2(board,player,x+1,y+1,rewards)
        elif board[x+1][y+1] == player:
            return False , 0 , (x,y)
        elif board[x+1][y+1] == 0:
            if rewards == 0:
                return False , 0 , (x,y)
            else:
                return True,rewards , (x+1 , y+1)
    else:
        return False , 0 , (x,y)

#%%            



#%%turning the opponent's pieces for a move
#check the direction vertical minus (down) on the ordinary viewed board
def vertical_minus_turn(board , player , x,y , turning_list , counter):
    if x < 7:
        if board[x+1][y] == -player:
            counter+=1
            turning_list.append((x+1,y))
            return vertical_minus_turn(board,player,x+1,y,turning_list, counter)
        elif board[x+1][y] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#check the direction vertical plus (up) on the ordinary viewed board
def vertical_plus_turn(board , player , x , y , turning_list,counter):
    if x > 0: 
        if board[x-1][y] == -player:
            counter+=1
            turning_list.append((x-1,y))
            return vertical_plus_turn(board,player,x-1,y,turning_list,counter)
        elif board[x-1][y] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#check the direction horizontal plus (right) on the ordinary viewed board
def horizontal_plus_turn(board , player , x , y , turning_list , counter):
    if y < 7:
        if board[x][y+1] == -player:
            counter+=1
            turning_list.append((x,y+1))
            return horizontal_plus_turn(board,player,x,y+1, turning_list , counter)
        elif board[x][y+1] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#check the direction horizontal minus (left) on the ordinary viewed board
def horizontal_minus_turn(board , player , x , y , turning_list , counter):
    if y > 0:
        if board[x][y-1] == -player:
            counter+=1
            turning_list.append((x,y-1))
            return horizontal_minus_turn(board,player,x,y-1,turning_list, counter)
        elif board[x][y-1] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#check the direction diagonal x=y to left on the ordinary viewed board            
def diagonal_minus_1_turn(board , player , x , y , turning_list , counter):
    if x < 7 and y > 0:
        if board[x+1][y-1] == -player:
            counter+=1
            turning_list.append((x+1,y-1))
            return diagonal_minus_1_turn(board,player,x+1,y-1,turning_list,counter)
        elif board[x+1][y-1] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#check the direction diagonal x=y to right on the ordinary viewed board                        
def diagonal_plus_1_turn(board , player , x , y , turning_list , counter):
    if x > 0 and y < 7:
        if board[x-1][y+1] == -player:
            counter+=1
            turning_list.append((x-1,y+1))
            return diagonal_plus_1_turn(board,player,x-1,y+1,turning_list, counter )
        elif board[x-1][y+1] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#check the direction diagonal x=-y to left on the ordinary viewed board            
def diagonal_plus_2_turn(board , player , x , y , turning_list , counter):
    
    if x > 0 and y > 0:
        if board[x-1][y-1] == -player:
            counter+=1
            turning_list.append((x-1,y-1))
            return diagonal_plus_2_turn(board,player,x-1,y-1, turning_list , counter)
        elif board[x-1][y-1] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#check the direction diagonal x=-y to right on the ordinary viewed board            
def diagonal_minus_2_turn(board , player , x , y , turning_list , counter):
    if x < 7 and y < 7:
        if board[x+1][y+1] == -player:
            counter+=1
            turning_list.append((x+1,y+1))
            return diagonal_minus_2_turn(board,player,x+1,y+1, turning_list , counter)
        elif board[x+1][y+1] == player:
            return turning_list
        else:
            for i in range(counter):
                turning_list.pop(-1)
            return turning_list
    else:
        return turning_list
#%%
        
#updates the v_matrix  
def valid_moves(board , player):
    avail = avail_matrix(8)
    result = np.where(board==player)
    listofcoordinates = list(zip(result[0],result[1]))
    for coordinate in listofcoordinates:
        x0,y0 = coordinate[0],coordinate[1]
        possibility , rewards , possible_coordinate = horizontal_plus(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
        possibility , rewards , possible_coordinate = horizontal_minus(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
        possibility , rewards , possible_coordinate = vertical_plus(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
        possibility , rewards , possible_coordinate = vertical_minus(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
        possibility , rewards , possible_coordinate = diagonal_plus_1(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
        possibility , rewards , possible_coordinate = diagonal_minus_1(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
        possibility , rewards , possible_coordinate = diagonal_plus_2(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
        possibility , rewards , possible_coordinate = diagonal_minus_2(board , player , x0 , y0 , 0)
        if possibility == True:
            avail[possible_coordinate] += rewards
    return avail

def get_turn_list(move , board , player):
    x,y = move[0],move[1]
    turn_list = []
    turn_list = vertical_minus_turn(board , player , x,y , turn_list , 0)
    turn_list = vertical_plus_turn(board , player , x,y , turn_list , 0)
    turn_list = horizontal_minus_turn(board , player , x,y , turn_list , 0)
    turn_list = horizontal_plus_turn(board , player , x,y , turn_list , 0)
    turn_list = diagonal_minus_1_turn(board , player , x,y , turn_list , 0)
    turn_list = diagonal_plus_1_turn(board , player , x,y , turn_list , 0)
    turn_list = diagonal_minus_2_turn(board , player , x,y , turn_list , 0)
    turn_list = diagonal_plus_2_turn(board , player , x,y , turn_list , 0)
    return turn_list






board = creat_board(8)
os.system('clear')
print(board)
while True:    
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
        
    print(board)
    player = 1
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