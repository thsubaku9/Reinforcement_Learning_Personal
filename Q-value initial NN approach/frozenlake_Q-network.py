import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.compat.v1.reset_default_graph()

learning_rate = tf.placeholder(tf.float16, shape=[])

inputState = tf.placeholder(dtype =tf.float16, shape = [1,16])
W = tf.Variable(tf.random.uniform([16,4], minval = 0, maxval = 0.2, dtype=tf.float16))
b = tf.Variable(tf.random.uniform([1,4], minval = -0.1, maxval = 0.1, dtype=tf.float16))

Qout = tf.add(tf.matmul(inputState,W),b)
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(dtype = tf.float16, shape = [1,4])
loss = tf.reduce_sum(tf.square(nextQ - Qout))

update = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

init = tf.initialize_all_variables()

discount_factor = 0.95
e = 0.2
epochs = 2000
init_learn_rate= 0.6
alpha = 0.1
jList = [];rList = []

sess = tf.Session()
sess.run(init)

for i in range(epochs):
    if(i%100 ==0):
        print("Epoch: "+ str(i))
    s = env.reset()
    rAll = 0
    destination = False
    curIter = 0;
    for curIter in range(100):
        a,allQ = sess.run([predict,Qout],feed_dict = {inputState : np.identity(16)[s:s+1]})

        if(np.random.rand(1) < e):  #promotes randomness for selection criteria
            a[0] = env.action_space.sample()

        s_next,reward,isDestination,additionalData = env.step(a[0])
        Q_next = sess.run(Qout, feed_dict = {inputState : np.identity(16)[s_next:s_next+1]})
        maxQ1 = np.max(Q_next)
        targetQ = allQ
        targetQ[0,a[0]] = reward + discount_factor*maxQ1
        _,W1 = sess.run([update,W],feed_dict = {inputState: np.identity(16)[s:s+1], nextQ: targetQ, learning_rate: init_learn_rate/(alpha*i + 1)})
        rAll += reward
        s = s_next
        if destination == True:
            e = 1/((i/50)+10)
            break
    jList.append(curIter)
    rList.append(rAll)
    
    

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

pl1 = plt.subplot(1,2,1)
pl2 = plt.subplot(1,2,2)
            
pl1.plot(rList)
plt2.plot(jList)

plt.show()
