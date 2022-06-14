import numpy as np
import sys
import csv
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
tf.random.set_seed(5)


class RRENetwork(object):
	# struct: dictionary with entry layers (list) and toggle (string)
	def __init__(self, psistruct, Kstruct, thetastruct, training_hp, pathname):
		self.pathname = pathname
		self.tfdtype = "float64"
		tf.keras.backend.set_floatx(self.tfdtype)
		self.tf_epochs = training_hp['epoch']
		self.tf_optimizer = tf.keras.optimizers.Adam()

		self.Psi = PsiNetwork(psistruct, self.pathname)
		self.K = KNetwork(Kstruct, self.pathname)
		self.Theta = ThetaNetwork(thetastruct, self.pathname)

	def rre_model(self, z_tf, t_tf):

		with tf.GradientTape(persistent=True) as tape:
			# Watching the two inputs we’ll need later, x and t
			tape.watch(z_tf)
			tape.watch(t_tf)

			# Packing together the inputs
			X_f = tf.stack([z_tf, t_tf],axis = 1)

			# Getting the prediction
			psi = self.Psi.net(X_f)
			log_h = tf.math.log(-psi)
			theta = self.Theta.net(-log_h)
			K = self.K.net(-log_h)
			# Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
			psi_z = tape.gradient(psi, z_tf)
			psi_t = tape.gradient(psi, t_tf)

			theta_t = tape.gradient(theta, t_tf)
			K_z = tape.gradient(K,z_tf)

		# Getting the other derivatives
		psi_zz = tape.gradient(psi_z, z_tf)

		f_residual =  theta_t - K_z*psi_z- K*psi_zz - K_z

		return psi, K, theta, f_residual

	def wrap_training_variables(self):
		psi_vars = self.Psi.net.trainable_variables
		K_vars = self.K.net.trainable_variables
		theta_vars = self.Theta.net.trainable_variables
		variables = psi_vars+K_vars+theta_vars
		return variables

	def loss(self, theta_pred, theta, f):
		l = tf.reduce_mean(tf.square(theta - theta_pred)) + \
			tf.reduce_mean(tf.square(f))
		return l

	def grad(self, z_tf, t_tf, theta):
		with tf.GradientTape() as tape:
			_, _, theta_pred, f = self.rre_model(z_tf, t_tf)
			loss_value = self.loss(theta_pred, theta, f)
		grads = tape.gradient(loss_value, self.wrap_training_variables())
		return loss_value, grads

	def tf_optimization(self, z_tf, t_tf, u):
		for epoch in range(self.tf_epochs):
			loss_value = self.tf_optimization_step(z_tf, t_tf, u)
			print("Epoch {0}, loss value: {1}\n".format(epoch, loss_value))
			if epoch %10 == 0 or epoch == self.tf_epochs:
				if epoch != 0:
					self.save_model()

	# @tf.function
	def tf_optimization_step(self, z_tf, t_tf, theta):
		loss_value, grads = self.grad(z_tf, t_tf, theta)
		self.tf_optimizer.apply_gradients(
				zip(grads, self.wrap_training_variables()))
		return loss_value

	def fit(self, z_tf, t_tf, theta):
		# self.logger.log_train_start(self)

		# Creating the tensors
		z_tf = tf.convert_to_tensor(z_tf, dtype=self.tfdtype)
		t_tf = tf.convert_to_tensor(t_tf, dtype=self.tfdtype)
		theta = tf.convert_to_tensor(theta, dtype=self.tfdtype)

		# Optimizing
		self.tf_optimization(z_tf, t_tf, theta)

		# self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

	def predict(self, z_tf, t_tf):
		z_tf = tf.convert_to_tensor(z_tf, dtype=self.tfdtype)
		t_tf = tf.convert_to_tensor(t_tf, dtype=self.tfdtype)
		psi, K, theta, _ = self.rre_model(z_tf, t_tf)
		return psi.numpy(), K.numpy(), theta.numpy()

	def save_model(self):
		self.Psi.save_model()
		self.K.save_model()
		self.Theta.save_model()

	def load_model(self):
		self.Psi.load_model()
		self.K.load_model()
		self.Theta.load_model()

class PsiNetwork(object):
	def __init__(self,psistruct, pathname):
		self.layers = psistruct['layers']
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.path = "{2}/Psi_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname)
		self.initialize_PsiNN()

	def initialize_PsiNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.PsiNN.add(tf.keras.layers.Lambda(
			# lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(PsiOutput(self.layers[-1]))

	def save_model(self):
		self.net.save(self.path+'.h5')

	def load_model(self):
		self.net.load_weights(self.path+'.h5')

class KNetwork(object):
	def __init__(self,Kstruct, pathname):
		self.layers = Kstruct['layers']
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.NN_toggle = Kstruct['toggle']
		self.path = "{2}/K{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		if self.NN_toggle == 'MNN':
			self.initialize_KMNN()
		else:
			self.initialize_KNN()

	def initialize_KMNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.KNN.add(tf.keras.layers.Lambda(
			# lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer=MNN_Init))
		self.net.add(KOutput(self.layers[-1]))

	def initialize_KNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.KNN.add(tf.keras.layers.Lambda(
			# lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(KOutput(self.layers[-1]))

	def save_model(self):
		self.net.save_weights(self.path+'.h5')

	def load_model(self):
		self.net.load_weights(self.path+'.h5')

class ThetaNetwork(object):
	def __init__(self,thetastruct, pathname):
		self.layers = thetastruct['layers']
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.NN_toggle = thetastruct['toggle']
		self.path = "{2}/Theta{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		if self.NN_toggle == 'MNN':
			self.initialize_ThetaMNN()
		else:
			self.initialize_ThetaNN()

	def initialize_ThetaNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.net.add(tf.keras.layers.Lambda(
			# lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(tf.keras.layers.Dense(
				self.layers[-1], activation=tf.nn.sigmoid,
				kernel_initializer="glorot_normal"))

	def initialize_ThetaMNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.net.add(tf.keras.layers.Lambda(
			# lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer=MNN_Init))
		self.net.add(tf.keras.layers.Dense(
				self.layers[-1], activation=tf.nn.sigmoid,
				kernel_initializer=MNN_Init))


	def save_model(self):
		self.net.save_weights(self.path+'.h5')

	def load_model(self):
		self.net.load_weights(self.path+'.h5')


class PsiOutput(tf.keras.layers.Layer):
	def __init__(self, units=32):
		super(PsiOutput, self).__init__()
		self.units = units

	def build(self, input_shape):
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="glorot_normal",
			trainable=True, name = "w")
		self.b = self.add_weight(
			shape=(self.units,), initializer="zeros", trainable=True, name = "b")

	def call(self, inputs):
		return -tf.math.exp(tf.matmul(inputs, self.w) + self.b)

	def get_config(self):
		return {"units": self.units}


class KOutput(tf.keras.layers.Layer):
	def __init__(self, units=32):
		super(KOutput, self).__init__()
		self.units = units

	def build(self, input_shape):
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=MNN_Init,
			trainable=True,name = "w")
		self.b = self.add_weight(
			shape=(self.units,), initializer="zeros", trainable=True, name = "b")

	def call(self, inputs):
		return tf.math.exp(tf.matmul(inputs, self.w) + self.b)

	def get_config(self):
		return {"units": self.units}


class MNN_Init(tf.keras.initializers.Initializer):

	def __init__(self):
		self.mean = 0 
		self.stddev = 0

	def __call__(self, shape, dtype=None, **kwargs):
		in_dim = shape[0]
		out_dim = shape[1]
		self.stddev = np.sqrt(2/(in_dim + out_dim))
		W = tf.random.normal(shape, stddev=self.stddev, dtype=dtype)
		return W**2

	def get_config(self):  # To support serialization
		return {"mean": self.mean, "stddev": self.stddev}


