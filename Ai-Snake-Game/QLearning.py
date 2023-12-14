#The q function
import random
import numpy as np


class QAgent:

    def __init__(self,actions):
        self.epsilon=1
        self.epsilon_decay_rate=0.95
        self.min_epsilon=0.001

        self.q=dict()


        self.learningRate=0.1
        self.actions=actions

        self.gamma=0.9
        



    def act(self,state):


        if state not in self.q:
            self.q[state]={a:0 for a in self.actions}


        if np.random.rand()<=self.epsilon:
            return np.random.choice(self.actions)

        qvalues=self.q[state]

        key,val=0,-100000

        for x in qvalues:
            xv=qvalues[x]
        
            if xv>val:
                key=x
                val=xv
        
        return key


    def updateAction(self,state,reward,action,nextState,done):
        if nextState not in self.q:
            self.q[nextState]={a:0 for a in self.actions}

        maxQ=max([self.q[nextState][a] for a in self.actions])
        self.q[state][action]+=self.learningRate*(reward+self.gamma*maxQ*(1-done)-self.q[state][action])
        
