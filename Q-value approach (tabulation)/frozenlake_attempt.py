import gym
import numpy as np


env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n,env.action_space.n])

lr = 0.8
y = 0.95
epochs = 2000
timePenalty = 0.09

rewardList = []

for i in range(epochs):
    s = env.reset()
    rAll = 0
    d = False
    for j in range(100):    
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)) -(timePenalty*j))
        s_next,r,d,_ = env.step(a)
        temporal_diff = (r+y*np.max(Q[s_next,:]))  - Q[s,a]
        Q[s,a] += lr*temporal_diff
        rAll += r
        s = s_next
        if d == True:
            break

    rewardList.append(rAll)

print("Score over time: " +  str(sum(rewardList)/epochs))

print(Q)
