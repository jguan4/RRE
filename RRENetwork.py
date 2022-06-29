import numpy as np
import math
import time
import sys
import csv
import pandas as pd
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import scipy.optimize as sopt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(5)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from RRETestProblem import RRETestProblem
from matplotlib.backends.backend_pdf import PdfPages


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
		self.psi_lb = tf.constant(self.training_hp['psi_lb'],dtype = self.tfdtype)
		self.psi_ub = tf.constant(self.training_hp['psi_ub'],dtype = self.tfdtype)
		self.Psi = PsiNetwork(psistruct, self.pathname)
		self.K = KNetwork(Kstruct, self.pathname)
		self.Theta = ThetaNetwork(thetastruct, self.pathname)
		self.weights = self.training_hp['weights']
		self.BCweight = self.convert_tensor(self.weights[0])
		self.fweight = self.convert_tensor(self.weights[1])
		self.thetaweight = self.convert_tensor(self.weights[2])
		self.fluxweight = self.convert_tensor(self.weights[3])
		self.loss_log = [['Epoch'],['l_theta'],['l_f'],['l_top'],['l_bottom']] # theta, f, top bound, lower bound
		self.log_time = time.time()
		self.wrap_variable_sizes()

	@tf.function
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
			flux = -K*(psi_z+1)
			psi_t = tape.gradient(psi, t_tf)

			theta_t = tape.gradient(theta, t_tf)
			K_z = tape.gradient(K,z_tf)

		# Getting the other derivatives
		psi_zz = tape.gradient(psi_z, z_tf)
		flux_z = tape.gradient(flux, z_tf)
		f_residual = theta_t + flux_z
		
		# f_residual =  theta_t - K_z*psi_z- K*psi_zz - K_z

		# flux = -K*(psi_z+1)

		# return psi, K, theta, f_residual, flux, psi_z
		return psi, K, theta, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz]

	# def loss(self, theta_pred, theta, f, log = False):
	# 	lf = self.loss_f(f)
	# 	ltheta = self.loss_reduce_mean(theta_pred,theta)
	# 	if log:
	# 		self.loss_log[1].append(ltheta.numpy())
	# 		self.loss_log[2].append(lf.numpy())
	# 	l = ltheta+lf*self.fweight
	# 		#+self.loss_flux(flux_pred, flux)\
	# 	return l

	def loss(self, theta_data, residual_data, boundary_data, log = False):
		loss = self.loss_theta(theta_data, log = log)*self.thetaweight+1e5*self.fweight*self.loss_residual(residual_data, log = log)+\
			self.BCweight*self.loss_boundary(boundary_data, log = log)
		return loss

	def loss_theta(self, theta_data, log = False):
		_, _, theta_pred, _, _, _ = self.rre_model(theta_data['z'], theta_data['t'])
		loss = self.loss_reduce_mean(theta_pred, theta_data['data'])
		if log:
			self.loss_log[1].append(loss.numpy())
		return loss 

	def loss_residual(self, residual_data, log = False):
		_, _, _, f_pred, _, _ = self.rre_model(residual_data['z'], residual_data['t'])
		loss = self.loss_f(f_pred)
		if log:
			self.loss_log[2].append(loss.numpy())
		return loss

	def loss_boundary(self, boundary_data, log = False):
		top_bound = boundary_data['top']
		bottom_bound = boundary_data['bottom']
		top_loss = self.loss_boundary_data(top_bound)
		bottom_loss = self.loss_boundary_data(bottom_bound)
		if log:
			self.loss_log[3].append(top_loss.numpy())
			self.loss_log[4].append(bottom_loss.numpy())
		return top_loss+bottom_loss

	def loss_boundary_data(self, bound):
		psi_pred, K_pred, theta_pred, f_pred, flux_pred, [psiz_pred, psit_pred, thetat_pred, Kz_pred, psizz_pred] = self.rre_model(bound['z'], bound['t'])
		if bound['type'] == 'flux':
			loss = self.loss_reduce_mean(flux_pred, bound['data'])*self.fluxweight*1e3
		elif bound['type'] == 'psiz':
			loss = self.loss_reduce_mean(psiz_pred, bound['data'])
		elif bound['type'] == 'psi':
			# loss = self.loss_reduce_mean(psi_pred, bound['data'])
			loss = self.loss_reduce_mean(psi_pred, bound['data'])/(self.psi_ub-self.psi_lb)**2
		return loss

	@tf.function
	def loss_reduce_mean(self, pred, data):
		l = tf.reduce_mean(tf.square(data - pred))
		return l 

	@tf.function
	def loss_f(self, f):
		l = tf.reduce_mean(tf.square(f))
		return l

	# @tf.function
	def grad(self, theta_data, residual_data, boundary_data, flatten = False):
		with tf.GradientTape() as tape:
			loss_value = self.loss(theta_data, residual_data, boundary_data, log = True)
			# loss_boundary = self.loss_boundary(boundary_data, log = True)
			# loss_value = loss+loss_boundary*self.BCweight
		grads = tape.gradient(loss_value, self.wrap_trainable_variables())
		return loss_value, grads

	def get_loss_and_flat_grad(self, theta_data, residual_data, boundary_data):
		def loss_and_flat_grad(w, log = False):
			with tf.GradientTape() as tape:
				self.set_weights(w)
				# _, _, theta_pred, f, _, _ = self.rre_model(z_tf, t_tf)
				# loss = self.loss(theta_pred, theta, f, log = log)
				# loss_boundary = self.loss_boundary(boundary_data, log = log)
				# loss_value = loss+loss_boundary*self.BCweight
				loss_value = self.loss(theta_data, residual_data, boundary_data, log = log)
			grad = tape.gradient(loss_value, self.wrap_trainable_variables())
			grad_flat = []
			for g in grad:
				grad_flat.append(tf.reshape(g, [-1]))
			grad_flat = tf.concat(grad_flat, 0)
			return loss_value, grad_flat.numpy()
		return loss_and_flat_grad	

	def fit(self, theta_data, residual_data, boundary_data):
		# self.logger.log_train_start(self)

		theta_data = self.convert_bound_data(theta_data)
		residual_data = self.convert_residual_data(residual_data)

		# Creating the tensors
		# z_tf = self.convert_tensor(z_tf)
		# t_tf = self.convert_tensor(t_tf)
		# theta = self.convert_tensor(theta)

		boundary_data['top'] = self.convert_bound_data(boundary_data['top'])
		boundary_data['bottom'] = self.convert_bound_data(boundary_data['bottom'])
	
		# Optimizing
		# self.tf_optimization(theta_data, residual_data, boundary_data)
		self.sopt_optimization(theta_data, residual_data, boundary_data)

	# def fit(self, z_tf, t_tf, theta, boundary_data):
	# 	# self.logger.log_train_start(self)

	# 	# Creating the tensors
	# 	z_tf = self.convert_tensor(z_tf)
	# 	t_tf = self.convert_tensor(t_tf)
	# 	theta = self.convert_tensor(theta)

	# 	boundary_data['top'] = self.convert_bound_data(boundary_data['top'])
	# 	boundary_data['bottom'] = self.convert_bound_data(boundary_data['bottom'])
	
	# 	# Optimizing
	# 	self.tf_optimization(z_tf, t_tf, theta, boundary_data)
	# 	self.sopt_optimization(z_tf, t_tf, theta, boundary_data)

	# 	# self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter)

	def tf_optimization(self, theta_data, residual_data, boundary_data):
		for epoch in range(self.tf_epochs):
			loss_value = self.tf_optimization_step(theta_data, residual_data, boundary_data)
			self.loss_log[0].append(epoch)
			print("Epoch {0}, loss value: {1}\n".format(epoch, loss_value))
			if epoch %50 == 0 or epoch == self.tf_epochs:
				# if epoch != 0:
				self.save_model()
				self.plotting(epoch)
				self.save_loss()

	def tf_optimization_step(self, theta_data, residual_data, boundary_data):
		loss_value, grads = self.grad(theta_data, residual_data, boundary_data)
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

	def sopt_optimization(self, theta_data, residual_data, boundary_data):
		x0 = self.get_weights()
		self.loss_and_flat_grad = self.get_loss_and_flat_grad(theta_data, residual_data, boundary_data)
		self.Nfeval = 0
		sopt.minimize(fun=self.loss_and_flat_grad, x0=x0, jac=True, method='L-BFGS-B', options = self.training_hp['lbfgs_options'], callback = self.sopt_callback)
		self.save_model()
		self.plotting('After LBFGS 1')

	def sopt_callback(self,Xi):
		self.loss_log[0].append(self.Nfeval)
		self.loss_and_flat_grad(Xi, log = True)
		if self.Nfeval %50 == 0:
			self.save_model()
			self.save_loss()
			self.plotting(self.Nfeval+self.tf_epochs)
		self.Nfeval += 1

	def predict(self, z_tf, t_tf):
		z_tf = self.convert_tensor(z_tf)
		t_tf = self.convert_tensor(t_tf)
		psi, K, theta, _, _, _ = self.rre_model(z_tf, t_tf)
		return psi.numpy(), K.numpy(), theta.numpy()

	def normalize_psi(self, psi):
		psi = (psi-self.psi_lb)/(self.psi_ub-self.psi_lb)
		return psi

	def denormalize_psi(self,psi):
		psi = psi*(self.psi_ub-self.psi_lb)+self.psi_lb
		return psi

	def save_model(self):
		self.Psi.save_model()
		self.K.save_model()
		self.Theta.save_model()
	
	def save_loss(self):
		np.savetxt("{0}/loss_{1}.csv".format(self.pathname,self.log_time),  np.column_stack(self.loss_log), delimiter =", ", fmt ='% s')

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

	def convert_residual_data(self, residual):
		residual['z'] = self.convert_tensor(residual['z'])
		residual['t'] = self.convert_tensor(residual['t'])
		return residual

	def plotting(self,epoch):
		if epoch == 0 or epoch == self.tf_epochs:
			if self.training_hp['csv_file'] is not None:
				data = pd.read_csv('./'+self.training_hp['csv_file'])
				t = data['time'].values[:,None]
				z = data['depth'].values[:,None]
				psi = data['head'].values[:,None]
				K = data['K'].values[:,None]
				C = data['C'].values[:,None]
				theta = data['theta'].values[:,None]
				flux = data['flux'].values[:,None]
				test_data = []
				T = None
				for item in [z,t,theta, K, psi]:
					Item = np.reshape(item,[251,1001])
					if T is None:
					# Items = Item[int(T/0.012),0:200]
						Items = Item[:,::20]
					else:
						Items = Item[int(T/0.012),::20]
					Itemt = np.reshape(Items,[np.prod(Items.shape),1])
					test_data.append(Itemt)
				self.ztest_whole, self.ttest_whole, self.thetatest_whole, self.Ktest_whole, self.psitest_whole = test_data
				test_data = []
				T = 0.6
				for item in [z,t,theta, K, psi]:
					Item = np.reshape(item,[251,1001])
					if T is None:
					# Items = Item[int(T/0.012),0:200]
						Items = Item[:,::20]
					else:
						Items = Item[int(T/0.012),::20]
					Itemt = np.reshape(Items,[np.prod(Items.shape),1])
					test_data.append(Itemt)
				self.ztest, self.ttest, self.thetatest, self.Ktest, self.psitest = test_data
				self.Nt = 251
				self.Nz = 51
			else:
				test_env = RRETestProblem(self.training_hp['dt'], self.training_hp['dz'], self.training_hp['T'], self.training_hp['Z'],self.training_hp['noise'], self.training_hp['name'],'')
				self.ttest_whole, self.ztest_whole, self.psitest_whole, self.Ktest_whole, self.thetatest_whole = test_env.get_training_data()

				test_env = RRETestProblem(self.training_hp['dt'], self.training_hp['dz'], 70, self.training_hp['Z'],self.training_hp['noise'], self.training_hp['name'],'')
				self.ttest, self.ztest, self.psitest, self.Ktest, self.thetatest = test_env.get_testing_data()
				self.Nt = int(self.training_hp['T']/self.training_hp['dt'])
				self.Nz = int(self.training_hp['Z']/self.training_hp['dz'])+1

			self.array_data = []

			for item in [self.ztest_whole, self.ttest_whole, self.thetatest_whole, self.Ktest_whole, self.psitest_whole]:
				Item = np.reshape(item,[self.Nt,self.Nz])
				self.array_data.append(Item)

			self.thetat_dis = (self.array_data[2][1:,:]-self.array_data[2][:-1,:])/self.training_hp['dt']
			self.psiz_dis = -(self.array_data[4][:,1:]-self.array_data[4][:,:-1])/self.training_hp['dz']
			self.psit_dis = (self.array_data[4][1:,:]-self.array_data[4][:-1,:])/self.training_hp['dt']
			self.Kz_dis = -(self.array_data[3][:,1:]-self.array_data[3][:,:-1])/self.training_hp['dz']
			self.psizz_dis = (self.psiz_dis[:,1:]-self.psiz_dis[:,:-1])/self.training_hp['dz']
			self.f_dis =  self.thetat_dis[:,1:-1] - self.Kz_dis[1:,1:]*self.psiz_dis[1:,1:]- (self.array_data[3][1:,1:-1])*self.psizz_dis[1:,:] - self.Kz_dis[1:,1:]

		array_data_temp = []

		psi_pred, K_pred, theta_pred, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz] = self.rre_model(self.convert_tensor(self.ztest_whole),self.convert_tensor(self.ttest_whole))
		for item in [f_residual, psi_z, psi_t, theta_t, K_z, psi_zz]:
			Item = np.reshape(item,[self.Nt,self.Nz])
			array_data_temp.append(Item)

		psi_pred, K_pred, theta_pred = self.predict(self.ztest,self.ttest)
		fig1, axs1 = plt.subplots(3, 2)
		#thetat
		axs1[0,0].plot(self.array_data[0][6,:], self.thetat_dis[6,:], 'b-')
		axs1[0,0].plot(self.array_data[0][6,:], array_data_temp[3][6,:], 'ro--')
		axs1[0,0].set_title('Theta_t vs z')
		axs1[0,0].set(xlabel='z', ylabel='theta_t')

		#psiz
		axs1[0,1].plot(self.array_data[0][6,1:], self.psiz_dis[6,:], 'b-')
		axs1[0,1].plot(self.array_data[0][6,1:], array_data_temp[1][6,1:], 'ro--')
		axs1[0,1].set_title('psi_z vs z')
		axs1[0,1].set(xlabel='z', ylabel='psi_z')

		#psit
		axs1[1,0].plot(self.array_data[0][6,:], self.psit_dis[6,:], 'b-')
		axs1[1,0].plot(self.array_data[0][6,:], array_data_temp[2][6,:], 'ro--')
		axs1[1,0].set_title('psi_t vs z')
		axs1[1,0].set(xlabel='z', ylabel='psi_t')

		#Kz
		axs1[1,1].plot(self.array_data[0][6,1:], self.Kz_dis[6,:], 'b-')
		axs1[1,1].plot(self.array_data[0][6,1:], array_data_temp[4][6,1:], 'ro--')
		axs1[1,1].set_title('K_z vs z')
		axs1[1,1].set(xlabel='z', ylabel='K_z')

		#psizz
		axs1[2,0].plot(self.array_data[0][6,1:-1], self.psizz_dis[6,:], 'b-')
		axs1[2,0].plot(self.array_data[0][6,1:-1], array_data_temp[5][6,1:-1], 'ro--')
		axs1[2,0].set_title('psi_zz vs z')
		axs1[2,0].set(xlabel='z', ylabel='psi_zz')

		#f
		axs1[2,1].plot(self.array_data[0][6,1:-1], self.f_dis[6,:], 'b-')
		axs1[2,1].plot(self.array_data[0][6,1:-1], array_data_temp[0][6,1:-1], 'ro--')
		axs1[2,1].set_title('f vs z')
		axs1[2,1].set(xlabel='z', ylabel='f')
		fig1.suptitle("epoch {0}".format(epoch), fontsize=16)
		# plt.show()

		# order_str = ['theta','K','psi']
		# for (ostr,data, pred) in zip(order_str,[thetatest_whole, Ktest_whole, psitest_whole], [theta_pred,K_pred,psi_pred]):
		# 	err = relative_error(data,pred)
		# 	print("For {0}, relative error is {1}.\n".format(ostr,err))

		# psi_pred, K_pred, theta_pred = rrenet.predict(ztest,ttest)

		fig, axs = plt.subplots(2, 2)
		axs[0,0].plot(self.ztest, theta_pred, 'ro--')
		axs[0,0].plot(self.ztest, self.thetatest, 'b-')
		axs[0,0].set_title('Theta vs z')
		axs[0,0].set(xlabel='z', ylabel='theta')

		axs[0,1].semilogy(self.ztest, K_pred, 'ro--')
		axs[0,1].semilogy(self.ztest, self.Ktest, 'b-')
		axs[0,1].set_title('K vs z')
		axs[0,1].set(xlabel='z', ylabel='K')

		axs[1,0].plot(self.ztest, psi_pred, 'ro--')
		axs[1,0].plot(self.ztest, self.psitest, 'b-')
		axs[1,0].set_title('Psi vs z')
		axs[1,0].set(xlabel='z', ylabel='psi')

		axs[1,1].plot(psi_pred, theta_pred, 'ro--')
		axs[1,1].plot(self.psitest, self.thetatest, 'b-')
		axs[1,1].set_title('Theta vs Psi')
		axs[1,1].set(xlabel='psi', ylabel='theta')
		fig.suptitle("T = {0}".format(self.ttest[0,0]), fontsize=16)

		# plt.show()
		filename = "{1}/epoch_{0}.pdf".format(epoch,self.pathname)
		pp = PdfPages(filename)
		fig_nums = plt.get_fignums()
		figs = [plt.figure(n) for n in fig_nums]
		for fig in figs:
			fig.savefig(pp, format='pdf')
		pp.close()
		plt.close('all')
		
		# self.save_multi_image(filename)
		# plt.savefig('my_plot.png')


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


