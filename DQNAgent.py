#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, MaxPooling2D, Conv2D, Reshape
from keras.optimizers import Adam

from keras import backend as K

def custom_activation(x):
    return (128 * K.sigmoid(x)) - 64

class DQNAgent:
    def __init__(self, env):
        self.memory = deque(maxlen=10000) # Memory size = 10000
        self.gamma = 1 #0.95    # discount rate
        # self.epsilon = 1.0  # exploration rate
        self.epsilon = 0.01
        # self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(env)        
        self.size = env.size
        self.actions_size = len(env.action_space)

    def _build_model(self, env):
        # Neural Net for Deep-Q learning Model
        
        nb_actions = len(env.action_space)
        state_shape = env.state_shape
        
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(env.size, env.size, 3), activation='relu')) # 3 filters? Parameters
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=state_shape, activation='relu')) # 3 filters? Parameters
        model.add(Conv2D(128, kernel_size=(2, 2), input_shape=state_shape, activation='relu'))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=state_shape, activation='relu'))
        model.add(Flatten())
        
        model.add(Dense(nb_actions, activation=custom_activation)) # last layer
        # model.add(Dense(nb_actions, activation='linear')) # last layer
        
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done, valid_actions_next):
        self.memory.append((np.array([state.T]), action, reward, np.array([next_state.T]), done, valid_actions_next))

    def act(self, state, valid_actions, eps_greedy=True):
        if eps_greedy and np.random.rand() <= self.epsilon: # Eps greedy
            return random.choice(valid_actions)
        act_values = self.model.predict(np.array([state.T]))[0,valid_actions]
        return valid_actions[np.argmax(act_values)]
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.size, self.size, 3))
        targets = np.zeros((batch_size, self.actions_size))
        
        count = 0
        
        for state, action, reward, next_state, done, valid_actions_next in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0, valid_actions_next]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            states[count] = state[0]
            targets[count] = target_f[0]
            count += 1
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# In[ ]:





# In[2]:


import gym
import gym_reversi

import tensorflow as tf
from keras import backend

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
backend.set_session(sess)


def opponent(board, avail):
    dummy=np.where(avail==np.amax(avail))
    # dummy=np.where(avail>0)
    maxavail = list(zip(dummy[0],dummy[1]))
    move = random.choice(maxavail)    
    return move


def self_play_opponent(board, avail):
    global env, agent
    state = env.get_state(-env.AI_Player)
    valid_actions = env.get_actions(-env.AI_Player)
    return env.action_ind_to_board_pos(agent.act(state, valid_actions))


env = gym.make("reversi-v0", opponent = opponent, AI_Player = 1)

agent = DQNAgent(env)


# state = env.reset()
# valid_actions = env.get_actions(env.AI_Player)
# 
# a = agent.act(state, valid_actions)
# next_state, r, done, _ = env.step(a)
# valid_actions_next = env.get_actions(env.AI_Player)
# agent.memorize(state, a, r, next_state, True, valid_actions_next)

# In[4]:


# Evaluate based on playing againts deterministic opponent
def test(num_games):
    global env, agent
    winner_counter = {
        "AI":0,
        "Tie":0,
        "Opponent":0
    }

    tmp = env.opponent
    env.opponent = opponent
    
    for g in range(num_games):
        done = False
        state = env.reset()
        while(not done):
            # env.render()
            valid_actions = env.get_actions(env.AI_Player)
            action = agent.act(state, valid_actions, eps_greedy=False)
            next_state, reward, done, _ = env.step(action)        
            state = next_state
        
        count = np.sum(env.board)
        if count == 0:
            winner = "Tie"
        elif env.AI_Player * count > 0:
            winner = "AI"
        else:
            winner = "Opponent"
        winner_counter[winner] += 1 / num_games
    env.opponent = tmp # reset opponent
    return winner_counter
test(50)


# In[ ]:


import time

batch_size = 32
GAMES_PER_EPISODE = 100
EPISODES = 100
win_rate = np.zeros(EPISODES)
tie_rate = np.zeros(EPISODES)

# env.opponent = self_play_opponent
env.opponent = opponent

for e in range(EPISODES):
    e_start = time.time()
    for g in range(GAMES_PER_EPISODE):
        done = False
        state = env.reset()
        while(not done):
            # env.render()
            valid_actions = env.get_actions(env.AI_Player)
            action = agent.act(state, valid_actions)

            next_state, reward, done, _ = env.step(action)        
            valid_actions_next = env.get_actions(env.AI_Player)
            agent.memorize(state, action, reward, next_state, done, valid_actions_next)
            state = next_state
            if(len(agent.memory) > batch_size):
                agent.replay(batch_size)
    e_duration = time.time() - e_start    
    winner_counter = test(50)
         
    print("episode: {}/{}, Time:{}: AI: {}, Tie: {}, Opponent: {}".format(e+1, EPISODES,e_duration,
        winner_counter["AI"], winner_counter["Tie"], winner_counter["Opponent"]))        
    win_rate[e] = winner_counter["AI"]
    tie_rate[e] = winner_counter["Tie"]
    
    if e % 100 == 0 and e > 0:
        agent.save("dqn{}_discount1_custAct.h5".format(e))
    
    env.AI_Player *= -1


# In[46]:


import matplotlib.pyplot as plt

plt.plot(range(1,EPISODES+1), win_rate, label="DQN")
plt.xlabel("Episode")
plt.ylabel("Win-rate")
plt.title("DQN")
plt.savefig("tmp2_dis95.png")


# In[ ]:


import pickle
pickle.dump( agent.memory, open( "agentMemory1000.p", "wb" ) )


# In[6]:


env.close()

