import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Bandit():
    def __init__(self,arms):
        self.arms = arms
        self.arm_prob = np.random.normal(0,1,[arms])


class ContextualBandits():
    def __init__(self,total,arms):
        self.state = 0
        self.total = total
        self.arms = arms
        self.bandits = []
        for x in range(self.total):
            self.bandits.append(Bandit(self.arms))

    def getBandit(self):
        self.state = np.random.randint(0,self.total)
        return self.state

    def pullArm(self,action):
        bandit = self.bandits[self.state,action]
        result = np.random.randn(1)
        if result >= bandit :
            return 1
        else:
            return -1

class agent():
    def __init__(self,lr,s_size,a_size):
        self.current_state = tf.placeholder(shape = [1],dtype = tf.int32)
        oneHot_current_state = tf.one_hot_encoding(self.current_state,s_size)
        

'''

tf.reset_default_graph()

init = tf.initialize_all_variables()
        
with tf.Session() as sess:

'''
