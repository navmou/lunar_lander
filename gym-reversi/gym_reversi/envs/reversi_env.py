import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pyglet
import pyglet.gl

class ReversiEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    UP = np.array((0,1))
    DOWN = np.array((0,-1))
    LEFT = np.array((-1,0))
    RIGHT = np.array((1,0))

    UP_RIGHT = np.array((1,1))
    DOWN_RIGHT = np.array((1,-1))
    UP_LEFT = np.array((-1,1))
    DOWN_LEFT = np.array((-1,-1))

    DIRECTIONS = [UP, DOWN, LEFT, RIGHT, UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT]
   
    board_size_px = 500
    bg_color = [0,200,0,255]
    highlight_color = [0,140,0,255]
    
    player_colors = {
        -1: [0,0,0,255], # black
        1: [255,255,255,255] # white 
    }
    
    circle_margin_px = 10

    cur_board = None
    board_grid_size = -1
    window = None
    
    INVALID_REWARD = -1e6
    
    
    def __init__(self, opponent, size = 8, AI_Player = 1):
        self.opponent = opponent
        self.size = size
        self.AI_Player = AI_Player
        self.square_width = -1
        st = self.reset()
        self.action_space = [i for i in range(60)]
        self.state_shape = st.shape
        
    def check_direction(self, player, position, direction, rewards):
        new_pos = tuple(position + direction)

        if (new_pos[0] >= 0 and new_pos[0] < self.size and
            new_pos[1] >= 0 and new_pos[1] < self.size):
            if self.board[new_pos] == -player:
                rewards += 1
                return self.check_direction(player, new_pos, direction, rewards)
            elif self.board[new_pos] == player:
                return False , 0 , position
            elif self.board[new_pos] == 0:
                if rewards != 0:
                    return True , rewards , new_pos
                else:
                    return False , 0 , position
        else:
            return False , 0 , position
    
    def get_turn_list_from_direction(self, player , position, direction, turning_list, counter):
        new_pos = tuple(position + direction)

        if (new_pos[0] >= 0 and new_pos[0] < self.size and
            new_pos[1] >= 0 and new_pos[1] < self.size):
            if self.board[new_pos] == -player:
                counter+=1
                turning_list.append(new_pos)
                return self.get_turn_list_from_direction(player , new_pos, direction, turning_list, counter)
            elif self.board[new_pos] == player:
                return turning_list
            else:
                for _ in range(counter):
                    turning_list.pop() # Remove last
                return turning_list
        else:
            for _ in range(counter):
                turning_list.pop()
            return turning_list
        
    def valid_moves(self, player):
        avail = np.zeros((self.size, self.size))
        result = np.where(self.board==player)
        
        for coordinate in zip(result[0],result[1]):
            for d in self.DIRECTIONS:
                possibility , rewards , possible_coordinate = self.check_direction(player, coordinate, d, 0)
                if possibility:
                    avail[possible_coordinate] += rewards
        return avail
    
    def get_turn_list(self, player, move):             
        turn_list = []
        for d in self.DIRECTIONS:
            turn_list = self.get_turn_list_from_direction(player , move, d, turn_list, 0)
        return turn_list
    
    def turn_from_move(self, player, move):
        self.board[move] = player
        for pos in self.get_turn_list(player, move):
            self.board[pos] = player
    
    def get_actions(self, player):
        actions = [self.board_pos_to_actions_ind(pos) for pos in np.argwhere(self.valid_moves(player) != 0)]
        return actions
        
    def board_pos_to_actions_ind(self, pos):
        middle = int(self.size/2)
        ig_ind1 = self.size * (middle-1) + middle - 1
        ig_ind2 = self.size * middle + middle
        
        ind = pos[0] * self.size + pos[1]
        return ind - 4 if ind > ig_ind2 else (ind - 2 if ind > ig_ind1 else ind)
        
    
    def action_ind_to_board_pos(self, action):
        middle = int(self.size/2)
        ig_ind1 = self.size * (middle-1) + middle - 1
        ig_ind2 = self.size * middle + middle
        
        ind = action + 4 if action > ig_ind2 - 4 else (action + 2 if action >= ig_ind1 else action)
        x = int(ind / self.size)
        y = int(ind - self.size * x)
        return (x,y)
    
    def get_state(self, player = None):
        if player == None:
            player = self.AI_Player
        st = np.zeros((3,self.size, self.size))
        st[0, self.valid_moves(player) > 0] = 1 
        st[1, self.board == player] = 1
        st[2, self.board == -player] = 1
        return st
    
    
    def step_new(self, action):        
        if not action in self.get_actions(self.AI_Player):
            return self.get_state(), self.INVALID_REWARD, False, { "INFO": "Invalid AI Move"} # True signals done
        
        x, y = self.action_ind_to_board_pos(action)
        self.turn_from_move(self.AI_Player, (x,y))
        
        while len(self.get_actions(-self.AI_Player)) > 0: # let opponent play until AI has move, or until it cannot move 
            avail = self.valid_moves(-self.AI_Player)
            pos = self.opponent(self.board, avail)

            if avail[pos] == 0:
                raise Exception("Invalid opponent move...", pos)

            self.turn_from_move(-self.AI_Player, pos)

            if len(self.get_actions(self.AI_Player)) > 0:
                return self.get_state(), 0, False, { "INFO": "AI Turn"} # True signals done
        if len(self.get_actions(self.AI_Player)) > 0:
            return self.get_state(), 0, False, { "INFO": "AI Turn" } # True signals done  
        
        sum_tiles = self.board.sum()
        finished_reward = 0 if sum_tiles == 0 else (1 if sum_tiles * self.AI_Player > 0 else -1)
        return self.get_state(), finished_reward, True, { "INFO": "Game done"} # True signals done    
      
                      
    
    def step(self, action):        
        if not action in self.get_actions(self.AI_Player):
            return self.get_state(), self.INVALID_REWARD, False, { "INFO": "Invalid AI Move"} # True signals done
        
        x, y = self.action_ind_to_board_pos(action)
        start_count = self.board.sum()
        self.turn_from_move(self.AI_Player, (x,y))
        
        while len(self.get_actions(-self.AI_Player)) > 0: # let opponent play until AI has move, or until it cannot move 
            avail = self.valid_moves(-self.AI_Player)
            pos = self.opponent(self.board, avail)

            if avail[pos] == 0:
                raise Exception("Invalid opponent move...", pos)

            self.turn_from_move(-self.AI_Player, pos)

            if len(self.get_actions(self.AI_Player)) > 0:
                return self.get_state(), self.AI_Player * (self.board.sum()- start_count), False, { "INFO": "AI Turn"} # True signals done
        if len(self.get_actions(self.AI_Player)) > 0:
            return self.get_state(), self.AI_Player * (self.board.sum()- start_count), False, { "INFO": "AI Turn" } # True signals done                      
        return self.get_state(), self.AI_Player * (self.board.sum()- start_count), True, { "INFO": "Game done"} # True signals done          
                                       

    def reset(self):
        self.board = np.zeros((self.size,self.size))
        middle = int(self.size/2)
        self.board[middle-1][middle-1], self.board[middle][middle] = -1,-1
        self.board[middle-1][middle], self.board[middle][middle-1] = 1,1
        
        if self.AI_Player == 1:
            return self.get_state()
        else:
            avail = self.valid_moves(-self.AI_Player)
            pos = self.opponent(self.board, avail)
            if avail[pos] == 0:
                raise Exception("Invalid opponent move...", pos)
            self.turn_from_move(-self.AI_Player, pos)
            
            return self.get_state()

    def render(self, mode='human'):
        if mode == 'human':
            if self.window is None:
                self.window = pyglet.window.Window(self.board_size_px, self.board_size_px)

                @self.window.event
                def on_close():
                    print("closing")
                    self.window.close()
                    self.window = None
                
                @self.window.event
                def on_window_close(window):
                    selfevent_loop.exit()    
                    
                @self.window.event
                def on_key_press(key, mod):
                    pass
            else:
                pyglet.clock.tick()
                self.window.clear()
            self.window.dispatch_events()
            
            scores = {
                -1: 0,
                 1: 0  
            }
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2i', [0, 0, self.board_size_px, 0, self.board_size_px, self.board_size_px, 0, self.board_size_px]), ('c4B',self.bg_color*4))
            
            square_width = self.board_size_px / self.size 
            avail = self.valid_moves(self.AI_Player)
            
            for i,j in np.argwhere(avail != 0):
                x0 = j * square_width
                x1 = (j+1) * square_width

                y0 = self.board_size_px - (i+1) * square_width
                y1 = self.board_size_px - i * square_width
                
                cs = [c for c in self.highlight_color]
                
                cs[2] = int(avail[(i,j)] / avail.max() * 100)
                                
                pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x0, y0, x1, y0, x1, y1, x0, y1]), ('c4B',cs*4))
            

            for i in range(self.size):
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES, 
                    ("v2f", (i * square_width, 0, i * square_width, self.board_size_px))
                )
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES, 
                    ("v2f", (0, i * square_width, self.board_size_px, i * square_width))
                )
          
            for i,j in np.argwhere(self.board != 0):
                col = self.player_colors[self.board[(i, j)]]

                scores[self.board[(i, j)]] += 1
                n = 20
                c_rad = (square_width - self.circle_margin_px) / 2 
                cx = (j + 0.5) * square_width
                cy = self.board_size_px - (i + 0.5) * square_width
                
                prev = [cx + c_rad * np.cos(0), cy + c_rad * np.sin(0)]
                
                
                for a in np.linspace(0, 2 * np.pi, n + 1)[1:]:    
                    new = [cx + c_rad * np.cos(a), cy + c_rad * np.sin(a)]
                    vertices = [cx, cy] + prev + new
                    pyglet.graphics.draw(3, pyglet.gl.GL_TRIANGLES, ('v2f', vertices), ('c4B',col * 3))
                    prev = new
            
            self.square_width = square_width
            self.window.set_caption("Deep Reversi    White: {}, Black: {}".format(scores[1],scores[-1]))
            self.window.flip()
                        

    def close(self):
        if not self.window is None:
            self.window.close()
            self.window = None