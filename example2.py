import tensorflow as tf
import numpy as np


#create data
x_data = np.random.rand(100)
y_data = x_data * 0.5 + 0.3

#create tensor
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros(1))

y = x_data * W + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(200):
    sess.run(train)
    if i % 10 == 0:
        print('step {}:W is {}, b is {}.'.format(i, sess.run(W), sess.run(b)))
