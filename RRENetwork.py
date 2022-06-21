import numpy as np
import math
import sys
import csv
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import scipy.optimize as sopt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(5)

# order to networks: Psi, K, Theta
class RRENetwork(object):
	# struct: dictionary with entry layers (list) and toggle (string)
	def __init__(self, psistruct, Kstruct, thetastruct, training_hp, pathname):
		self.pathname = pathname
		self.norm = training_hp['norm']
		self.tfdtype = "float64"
		tf.keras.backend.set_floatx(self.tfdtype)
		self.training_hp = training_hp
		self.adam_options = self.training_hp['adam_options']
		self.tf_epochs = self.adam_options['epoch']
		self.tf_optimizer = tf.keras.optimizers.Adam()
		self.lb = tf.constant(self.training_hp['lb'],dtype = self.tfdtype)
		self.ub = tf.constant(self.training_hp['ub'],dtype = self.tfdtype)
		self.Psi = PsiNetwork(psistruct, self.pathname)
		self.K = KNetwork(Kstruct, self.pathname)
		self.Theta = ThetaNetwork(thetastruct, self.pathname)
		self.weight = self.convert_tensor(training_hp['weight'])
		self.wrap_variable_sizes()

	def rre_model(self, z_tf, t_tf):

		with tf.GradientTape(persistent=True) as tape:
			# Watching the two inputs we’ll need later, x and t
			tape.watch(z_tf)
			tape.watch(t_tf)

			# Packing together the inputs
			X_f = tf.squeeze(tf.stack([z_tf, t_tf],axis = 1))
			if self.norm == '_norm':
				X_f = 2*(X_f - self.lb)/(self.ub - self.lb) -1
			elif self.norm == '_norm1':
				X_f = (X_f - self.lb)/(self.ub - self.lb)

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
		print(psi_z)
		print(psi_zz)
		input()
		f_residual =  theta_t - K_z*psi_z- K*psi_zz - K_z

		flux = -K*(psi_z+1)

		return psi, K, theta, f_residual, flux, psi_z

	def wrap_trainable_variables(self):
		psi_vars = self.Psi.net.trainable_variables
		K_vars = self.K.net.trainable_variables
		theta_vars = self.Theta.net.trainable_variables
		variables = psi_vars+K_vars+theta_vars
		return variables

	def wrap_variable_sizes(self):
		psi_w = self.Psi.size_w
		psi_b = self.Psi.size_b
		K_w = self.K.size_w
		K_b = self.K.size_b
		theta_w = self.Theta.size_w
		theta_b = self.Theta.size_b
		self.sizes_w = psi_w+K_w+theta_w
		self.sizes_b = psi_b+K_b+theta_b

	def wrap_layers(self):
		psi_layers = self.Psi.net.layers
		K_layers = self.K.net.layers
		theta_layers = self.Theta.net.layers
		layers = psi_layers+K_layers+theta_layers
		return layers

	def get_weights(self, convert_to_tensor=True):
		w = []
		for layer in self.wrap_layers():
			weights_biases = layer.get_weights()
			weights = weights_biases[0].flatten()
			biases = weights_biases[1]
			w.extend(weights)
			w.extend(biases)
		if convert_to_tensor:
			w = self.convert_tensor(w)
		return w

	def set_weights(self, w):
		for i, layer in enumerate(self.wrap_layers()):
			start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
			end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
			weights = w[start_weights:end_weights]
			w_div = int(self.sizes_w[i] / self.sizes_b[i])
			weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
			biases = w[end_weights:end_weights + self.sizes_b[i]]
			weights_biases = [weights, biases]
			layer.set_weights(weights_biases)

	def get_loss_and_flat_grad(self, z_tf, t_tf, theta, boundary_data):
		def loss_and_flat_grad(w):
			with tf.GradientTape() as tape:
				self.set_weights(w)
				_, _, theta_pred, f, _, _ = self.rre_model(z_tf, t_tf)
				loss = self.loss(theta_pred, theta, f)
				loss_boundary = self.loss_boundary(boundary_data)
				loss_value = loss+loss_boundary*self.weight
			grad = tape.gradient(loss_value, self.wrap_trainable_variables())
			grad_flat = []
			for g in grad:
				grad_flat.append(tf.reshape(g, [-1]))
			grad_flat = tf.concat(grad_flat, 0)
			return loss_value, grad_flat.numpy()
		return loss_and_flat_grad	

	def loss(self, theta_pred, theta, f):
		l = self.loss_reduce_mean(theta_pred,theta)+self.loss_f(f)
			#+self.loss_flux(flux_pred, flux)\
		return l

	def loss_reduce_mean(self, pred, data):
		l = tf.reduce_mean(tf.square(data - pred))
		return l 

	def loss_f(self, f):
		l = tf.reduce_mean(tf.square(f))
		return l

	def loss_boundary(self, boundary_data):
		top_bound = boundary_data['top']
		bottom_bound = boundary_data['bottom']
		top_loss = self.loss_boundary_data(top_bound)
		bottom_loss = self.loss_boundary_data(bottom_bound)
		return top_loss+bottom_loss

	def loss_boundary_data(self, bound):
		psi_pred, K_pred, theta_pred, f_pred, flux_pred, psiz_pred = self.rre_model(bound['z'], bound['t'])
		if bound['type'] == 'flux':
			loss = self.loss_reduce_mean(flux_pred, bound['data'])
		elif bound['type'] == 'psiz':
			loss = self.loss_reduce_mean(psiz_pred, bound['data'])
		elif bound['type'] == 'psi':
			loss = self.loss_reduce_mean(psi_pred, bound['data'])
		return loss

	def grad(self, z_tf, t_tf, theta, boundary_data, flatten = False):
		with tf.GradientTape() as tape:
			_, _, theta_pred, f, _, _ = self.rre_model(z_tf, t_tf)
			loss = self.loss(theta_pred, theta, f)
			loss_boundary = self.loss_boundary(boundary_data)
			loss_value = loss+loss_boundary*self.weight
		grads = tape.gradient(loss_value, self.wrap_trainable_variables())
		return loss_value, grads

	def tf_optimization(self, z_tf, t_tf, theta, boundary_data):
		for epoch in range(self.tf_epochs):
			loss_value = self.tf_optimization_step(z_tf, t_tf, theta, boundary_data)
			print("Epoch {0}, loss value: {1}\n".format(epoch, loss_value))
			if epoch %50 == 0 or epoch == self.tf_epochs:
				if epoch != 0:
					self.save_model()

	def tf_optimization_step(self, z_tf, t_tf, theta, boundary_data):
		loss_value, grads = self.grad(z_tf, t_tf, theta, boundary_data)
		self.tf_optimizer.apply_gradients(
					zip(grads, self.wrap_trainable_variables()))
		return loss_value

	# @tf.function
	# def tf_optimization_step_batch(self, z_tf, t_tf, theta, zb_tf, tb_tf, flux, batch = 1024):
	# 	N = math.ceil(tf.size(z_tf)/batch)
	# 	loss_value = 0.0
	# 	gs = [tf.zeros(tf.shape(v), dtype=v.dtype) for v in self.wrap_trainable_variables()]
	# 	for n in range(N): 
	# 		inds = n*batch
	# 		# inde = (n+1)*batch
	# 		inde = tf.size(z_tf) if n == N-1 else (n+1)*batch
	# 		loss_val, grads = self.grad(z_tf[inds:inde], t_tf[inds:inde], theta[inds:inde], flux[inds:inde])
	# 		for i in range(len(grads)):
	# 			gs[i] = gs[i]+grads[i]
	# 		loss_value = loss_value+loss_val
	# 	self.tf_optimizer.apply_gradients(
	# 				zip(gs, self.wrap_trainable_variables()))
	# 	return loss_value

	def sopt_optimization(self, z_tf, t_tf, theta, boundary_data):
		x0 = self.get_weights()
		loss_and_flat_grad = self.get_loss_and_flat_grad(z_tf, t_tf, theta, boundary_data)
		sopt.minimize(fun=loss_and_flat_grad, x0=x0, jac=True, method='L-BFGS-B', options = self.training_hp['lbfgs_options'])
		self.save_model()

	def fit(self, z_tf, t_tf, theta, boundary_data):
		# self.logger.log_train_start(self)

		# Creating the tensors
		z_tf = self.convert_tensor(z_tf)
		t_tf = self.convert_tensor(t_tf)
		theta = self.convert_tensor(theta)

		boundary_data['top'] = self.convert_bound_data(boundary_data['top'])
		boundary_data['bottom'] = self.convert_bound_data(boundary_data['bottom'])
	
		# Optimizing
		self.tf_optimization(z_tf, t_tf, theta, boundary_data)
		self.sopt_optimization(z_tf, t_tf, theta, boundary_data)

		# self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

	def predict(self, z_tf, t_tf):
		z_tf = self.convert_tensor(z_tf)
		t_tf = self.convert_tensor(t_tf)
		psi, K, theta, _, _, _ = self.rre_model(z_tf, t_tf)
		return psi.numpy(), K.numpy(), theta.numpy()

	def save_model(self):
		self.Psi.save_model()
		self.K.save_model()
		self.Theta.save_model()

	def load_model(self):
		self.Psi.load_model()
		self.K.load_model()
		self.Theta.load_model()

	def convert_tensor(self, data):
		data_tf = tf.convert_to_tensor(data, dtype=self.tfdtype)
		return data_tf

	def convert_bound_data(self, bound):
		bound['z'] = self.convert_tensor(bound['z'])
		bound['t'] = self.convert_tensor(bound['t'])
		bound['data'] = self.convert_tensor(bound['data'])
		return bound

class PsiNetwork(object):
	def __init__(self,psistruct, pathname):
		self.layers = psistruct['layers']
		# self.lb = tf.constant(psistruct['lb'],dtype = tf.float64)
		# self.ub = tf.constant(psistruct['ub'],dtype = tf.float64)
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.size_w = []
		self.size_b = []
		self.path = "{2}/Psi_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname)
		self.initialize_PsiNN()
		self.get_weights_struct()

	def initialize_PsiNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.net.add(tf.keras.layers.Lambda(
		# 	lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(PsiOutput(self.layers[-1]))

	def get_weights_struct(self):
		for i in range(len(self.net.layers)):
			self.size_w.append(tf.size(self.net.layers[i].kernel))
			self.size_b.append(tf.size(self.net.layers[i].bias))

	def save_model(self):
		self.net.save(self.path+'.h5')

	def load_model(self):
		self.net.load_weights(self.path+'.h5')

class KNetwork(object):
	def __init__(self,Kstruct, pathname):
		self.layers = Kstruct['layers']
		self.size_w = []
		self.size_b = []
		# self.lb = psistruct['lb']
		# self.ub = psistruct['ub']
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.NN_toggle = Kstruct['toggle']
		self.path = "{2}/K{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		if self.NN_toggle == 'MNN':
			self.initialize_KMNN()
		else:
			self.initialize_KNN()
		self.get_weights_struct()

	def get_weights_struct(self):
		for i in range(len(self.net.layers)):
			self.size_w.append(tf.size(self.net.layers[i].kernel))
			self.size_b.append(tf.size(self.net.layers[i].bias))

	def initialize_KMNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.net.add(tf.keras.layers.Lambda(
		# 	lambda X: 2.0*(X - self.lb)/(ub - self.lb) - 1.0))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer=MNN_Init))
		self.net.add(KOutput(self.layers[-1]))

	def initialize_KNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.net.add(tf.keras.layers.Lambda(
		# 	lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
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
		self.size_w = []
		self.size_b = []
		# self.lb = psistruct['lb']
		# self.ub = psistruct['ub']
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.NN_toggle = thetastruct['toggle']
		self.path = "{2}/Theta{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		if self.NN_toggle == 'MNN':
			self.initialize_ThetaMNN()
		else:
			self.initialize_ThetaNN()
		self.get_weights_struct()

	def get_weights_struct(self):
		for i in range(len(self.net.layers)):
			self.size_w.append(tf.size(self.net.layers[i].kernel))
			self.size_b.append(tf.size(self.net.layers[i].bias))

	def initialize_ThetaNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		# self.net.add(tf.keras.layers.Lambda(
		# 	lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
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
		# 	lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
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
		self.kernel = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="glorot_normal",
			trainable=True, name = "kernel")
		self.bias = self.add_weight(
			shape=(self.units,), initializer="zeros", trainable=True, name = "bias")

	def call(self, inputs):
		return -tf.math.exp(tf.matmul(inputs, self.kernel) + self.bias)

	def get_config(self):
		return {"units": self.units}


class KOutput(tf.keras.layers.Layer):
	def __init__(self, units=32):
		super(KOutput, self).__init__()
		self.units = units

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=MNN_Init,
			trainable=True,name = "kernel")
		self.bias = self.add_weight(
			shape=(self.units,), initializer="zeros", trainable=True, name = "bias")

	def call(self, inputs):
		return tf.math.exp((tf.matmul(inputs, self.kernel) + self.bias))

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


