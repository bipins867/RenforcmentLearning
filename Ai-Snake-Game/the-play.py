from QLearning import QAgent
from snake_env import SnakeEnv
import numpy as np
import pygame
import sys


pygame.init()
env=SnakeEnv()
agent=QAgent(env.action_space)


for _ in range(1000):
    state=env.reset()
    done=False
    while not done:

        for event in pygame.event.get():

            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()

        env.render()
        action=agent.act(state)
        nextState,reward,done=env.step(action)
        agent.updateAction(state,reward,action,nextState,done)
        state=nextState
    print("Episode : ",_,"\tScore : ",env.score)
    agent.epsilon=max(agent.epsilon*agent.epsilon_decay_rate,agent.min_epsilon)
    #Epsilon update policy
    

    
