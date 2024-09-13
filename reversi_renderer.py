from tkinter import *     
import numpy as np
import time

_board_size_px = 500
_bg_color = "green"
_highlight_color = "red"
_line_color = "white"

_player_1_color = "white"
_player_2_color = "black"

_circle_margin_px = 5

_cur_board = None
_board_grid_size = -1
_square_width = -1

clicked_event = None
timer_event = None
timer_pause = 100 # ms

def paint_board(board_mat, highlight_pos = None):
    global _cur_board, _square_width, _board_grid_size
    _cur_board = board_mat.copy()
    
    _board.create_rectangle(0, 0, _board_size_px, _board_size_px, fill = _bg_color)
    _board_grid_size = _cur_board.shape[0]
    _square_width = _board_size_px / _board_grid_size
    
    for i in range(_board_grid_size):
        _board.create_line(i * _square_width, 0, i * _square_width, _board_size_px, fill=_line_color)
        _board.create_line(0, i * _square_width, _board_size_px, i * _square_width, fill=_line_color)
        
    scores = {
        _player_1_color: 0,
        _player_2_color: 0
    }
    
    
    if highlight_pos != None:
        i = highlight_pos[0]
        j = highlight_pos[1]
        
        _board.create_line(i * _square_width, j * _square_width, i * _square_width, (j + 1)* _square_width, fill=_highlight_color, width =  5)
        _board.create_line((i+1) * _square_width, j * _square_width, (i+1) * _square_width, (j + 1)* _square_width, fill=_highlight_color, width =  5)
        _board.create_line(i * _square_width, j * _square_width, (i + 1) * _square_width, j * _square_width, fill=_highlight_color, width =  5)
        _board.create_line(i * _square_width, (j+1) * _square_width, (i + 1) * _square_width, (j+1) * _square_width, fill=_highlight_color, width =  5)
        
        
    for i,j in np.argwhere(board_mat != 0):
        x0 = i * _square_width + _circle_margin_px
        x1 = (i+1) * _square_width - _circle_margin_px

        y0 = j * _square_width + _circle_margin_px
        y1 = (j+1) * _square_width - _circle_margin_px

        col = _player_1_color if board_mat[(i, j)] == 1 else _player_2_color
        
        scores[col] += 1
        
        _board.create_oval(x0, y0, x1, y1 , fill = col)
        
    _score_lab.config(text="White: {}, Black: {}".format(scores[_player_1_color],scores[_player_2_color]))
        
        
def _board_click(event):
    i = (int) (event.x / _square_width)
    j = (int) (event.y / _square_width)
    
    if i < _board_grid_size and j < _board_grid_size and clicked_event != None:
        clicked_event((i,j))
def Close():
    _root.destroy()
    

def _timer_tick():
    if timer_event != None:
        timer_event()
    _root.after(timer_pause,_timer_tick)

def Start(init_mat):
    global _root, _board, _score_lab
    
    _root = Tk()
    _root.resizable(False, False)
    _root.title("Deep Reversi")
    _board = Canvas(_root, 
               width=_board_size_px, 
               height=_board_size_px)
    _board.pack(expand = YES, fill = BOTH)
    _board.bind("<Button-1>", _board_click)

    _score_lab = Label(_root, text = "White: {}, Black: {}".format(0,0), font=("Helvetica", 16))
    _score_lab.pack(side = BOTTOM)
    
    paint_board(init_mat)
    if timer_event != None:
        _root.after(timer_pause,_timer_tick)
    mainloop()