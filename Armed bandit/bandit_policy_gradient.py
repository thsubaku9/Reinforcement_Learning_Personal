import tensorflow as tf
import numpy as np

total_bandits = 9

bandits = np.random.gamma(scale = 9, shape = 5, size = [1,total_bandits])
bandits -= bandits.mean(axis = 1)
bandits /= np.sqrt(bandits.var(axis = 1))

def pullBandit(bandit):
    random_result = np.random.randn(1)
    if random_result > bandit:
        return 1
    else:
        return -1


tf.compat.v1.reset_default_graph()

W = tf.Variable(tf.ones([total_bandits]))
actionNext = tf.argmax(W,0)


reward_holder = tf.placeholder(shape =[1], dtype = tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(W,action_holder,[1])

loss = -(tf.math.log(responsible_weight)*reward_holder)
update = tf.train.AdamOptimizer(learning_rate = 0.002, beta1 = 0.3).minimize(loss)
randomness_selection = 0.8

init = tf.initialize_all_variables()    

total_iterations = 1000
total_reward = np.zeros(total_bandits)

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_iterations:
        i+=1
        if np.random.rand(1) < randomness_selection:
            action = np.random.randint(total_bandits)
            #randomness_selection is not degraded over here to ensure that solution doesn't get stuck in a local minima
        else:
            action = sess.run(actionNext)
        reward = pullBandit(bandits[0][action])

        cur_loss,resp,ww = sess.run([update,responsible_weight,W],feed_dict = {reward_holder: [reward], action_holder: [action]})
        total_reward[action] += reward
        if(i%100 ==0):
            print(str(total_reward))

print("best bandit so far is" + str(np.argmax(ww)+1))

if (np.argmax(ww) == np.argmin(np.array(bandits))):
    print("YEP")
else:
    print("Nope!")
      
                                    
