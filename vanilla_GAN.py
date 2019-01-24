# Copyright (c) 2019 by huyz. All Rights Reserved.
# Reference: Generative Adversarial Nets

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST', one_hot=True)

num_epochs = 1000
batch_size=128
learning_rate = 0.0001
num_hidden = 128
input_dim = 784
noise_dim = 128

X = tf.placeholder(tf.float32, shape=[None, input_dim])
Z = tf.placeholder(tf.float32, shape=[None, noise_dim])

def generator(inputs, reuse=None, scope='generator'):
	with tf.variable_scope(scope, reuse=reuse):
		with slim.arg_scope([slim.fully_connected], num_outputs=num_hidden, activation_fn=tf.nn.relu):

			net = slim.fully_connected(inputs, scope='fc1')
			output = slim.fully_connected(net, num_outputs=input_dim, activation_fn=tf.nn.sigmoid, scope='output')

			return output

def discriminator(inputs, reuse=None, scope='discriminator'):
	with tf.variable_scope(scope, reuse=reuse) as scope:
		with slim.arg_scope([slim.fully_connected], num_outputs=num_hidden, activation_fn=tf.nn.relu):

			if reuse:
				scope.reuse_variables()

			net = slim.fully_connected(inputs, scope='fc1')
			output = slim.fully_connected(net, num_outputs=1, activation_fn=tf.nn.sigmoid, scope='output')

			return output

def get_noise(batch_size, noise_dim):
	return np.random.normal(size=[batch_size, noise_dim])

# Generate images using random noise
G = generator(Z)

# Return a value for thr real image
D_real = discriminator(X)
# Return a value for a generated image
D_fake = discriminator(G, reuse=True)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))
tf.summary.scalar('loss_D', -loss_D)

loss_G = tf.reduce_mean(tf.log(D_fake))
tf.summary.scalar('loss_G', -loss_G)

# tf.get_collection(key, scope).  # 从一个集合(collection)中取出全部变量组成一个list
# tf.GraphKeys.TRAINABLE_VARIABLES 表示可学习的变量
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_G, var_list=vars_G)

print('Start learning...')

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('./log', sess.graph)

	total_batch = int(mnist.train.num_examples / batch_size)
	loss_val_D, loss_val_G = 0, 0

	for epoch in range(num_epochs):
		for i in range(total_batch):
			batch_xs, _ = mnist.train.next_batch(batch_size)
			noise = get_noise(batch_size, noise_dim)

			_, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
			_, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

		summary = sess.run(merged, feed_dict={X: batch_xs, Z: noise})
		writer.add_summary(summary, global_step=epoch)

		print('Epoch: {:0>3d}, D_loss: {:.4f}, G_loss: {:.4f}'.format(epoch+1, -loss_val_D, -loss_val_G))

		if epoch == 0 or epoch % 10 == 0 or epoch == num_epochs - 1:
			sample_size = 10
			noise = get_noise(batch_size, noise_dim)
			samples = sess.run(G, feed_dict={Z: noise})

			fig, ax = plt.subplots(nrows=1, ncols=sample_size, figsize=(sample_size, 1))
			for i in range(sample_size):
				ax[i].set_axis_off()
				ax[i].imshow(np.reshape(samples[i], (28, 28)))

			if not os.path.exists('./samples'):
				try:
					os.makedirs('./samples')
				except OSError:
					pass
			plt.savefig('samples/{}.png'.format(str(epoch).zfill(3), bbox_inches='tight'))
			plt.close(fig)

	print('Learning finished!')