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
from RRENetwork import *
from RRENetwork_flux import *
from matplotlib.backends.backend_pdf import PdfPages
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

class RRENetwork_flux_ide(RRENetwork_flux):
	def __init__(self, psistruct, Kstruct, thetastruct, training_hp, pathname):
		super().__init__(psistruct, Kstruct, thetastruct, training_hp, pathname)
		# Defining the two additional trainable variables for identification
		# self.thetas = tf.Variable([2.0], dtype=self.tfdtype)
		# self.thetar = tf.Variable([-5.0], dtype=self.tfdtype)
		# self.n = tf.Variable([0.0], dtype=self.tfdtype)
		# self.alpha = tf.Variable([-10.0], dtype=self.tfdtype)
		# self.Ks = tf.Variable([3.0], dtype=self.tfdtype)
		# self.l = tf.Variable([-1.0], dtype=self.tfdtype)

	# @tf.function
	def rre_model(self, z_tf, t_tf, flux_tf):
		# thetas, thetar, n, alpha, Ks = self.get_params()
		thetas  = 0.41
		thetar = 0.065
		n = 1.89
		alpha = 0.075
		Ks = 106.1

		with tf.GradientTape(persistent=True) as tape:
			# Watching the two inputs we’ll need later, x and t
			tape.watch(z_tf)
			tape.watch(t_tf)
			tape.watch(flux_tf)

			# Packing together the inputs
			X_f = tf.squeeze(tf.stack([z_tf, t_tf, flux_tf],axis = 1))
			if self.norm == '_norm':
				X_f = 2*(X_f - self.lb)/(self.ub - self.lb) -1
			elif self.norm == '_norm1':
				X_f = (X_f - self.lb)/(self.ub - self.lb)

			# Getting the prediction
			psi = self.Psi.net(X_f)
			# psi = -psiout

			log_h = tf.math.log(-psi)
			theta = self.Theta.net(-log_h)
			K = self.K.net(log_h)

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
		flux_residual = flux-flux_tf
		del tape

		m = 1-1/n
		theta_residual = theta - (thetar + (thetas-thetar)/(1+(-alpha*psi)**n)**m)
		Se = (theta-thetar)/(thetas-thetar)
		K_residual = K/(Ks*(1-(-alpha*psi)**(n-1)*(1+(-alpha*psi)**n)**(-m))**2/(1+(-alpha*psi)**n)**(m/2))-1
		# K_residual = K - (Ks*(Se**0.5)*(1-(1-Se**(1/m))**m)**2)
		
		# print(Se)
		# print(Se**(1/m))
		# print((1-(1-Se**(1/m))**m))
		# print(tf.math.abs(alpha*psi))
		# print(tf.math.abs(alpha*psi)**n)
		# print(1-tf.math.abs(alpha*psi)**n)
		# print((1-tf.math.abs(alpha*psi)**n)**m)
		# input()
		# print(theta_residual)
		# print(K_residual)

		# f_residual =  theta_t - K_z*psi_z- K*psi_zz - K_z

		# flux = -K*(psi_z+1)

		# return psi, K, theta, f_residual, flux, psi_z
		return psi, K, theta, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual, theta_residual, K_residual]

	def loss_theta(self, theta_data, log = False):
		_, _, theta_pred, _, _, _ = self.rre_model(theta_data['z'], theta_data['t'], theta_data['flux'])
		loss = self.loss_reduce_mean(theta_pred, theta_data['data'])
		if log:
			self.loss_log[1].append(loss.numpy())
		return loss

	def loss_residual(self, residual_data, log = False, func_residual_toggle = False):
		# with tf.GradientTape() as tape:
		_, _, _, f_pred, _, [_, _, _, _, _, _, theta_residual, K_residual] = self.rre_model(residual_data['z'], residual_data['t'], residual_data['flux'])
		res = self.loss_f(f_pred)
		func_residuals = self.loss_f(K_residual[tf.math.is_finite(K_residual)])+self.loss_f(theta_residual[tf.math.is_finite(theta_residual)])
		# func_residuals = self.loss_f(K_residual[tf.math.is_finite(K_residual)])
		# func_residuals = self.loss_f(theta_residual[tf.math.is_finite(theta_residual)])
		# print(tape.gradient(func_residuals,self.K.net.trainable_variables))
		# input()
		print(func_residuals)
		# if func_residual_toggle:
			# if math.isnan(func_residuals) or func_residuals.numpy()>1e0:
				# loss = res 
			# else:
		loss = res+ 1e-5*func_residuals
		# else:
			# loss = res
			# loss = self.loss_f(f_pred)+(func_residuals)*1e-3
		if log:
			self.loss_log[2].append(loss.numpy())
		return loss

	def loss_residual_batch(self, residual_data, log = False, batch = 4096):
		N = len(residual_data['z'])
		if N>batch:
			numbatch = int(np.ceil(N/batch))
			loss = 0.0
			for i in range(numbatch):
				startind = i*batch
				endind = tf.math.minimum((i+1)*batch,N)
				zb = residual_data['z'][startind:endind]
				tb = residual_data['t'][startind:endind]
				fluxb = residual_data['flux'][startind:endind]
				_, _, _, f_pred, _, [_, _, _, _, _, _, theta_residual, K_residual] = self.rre_model(zb, tb, fluxb)
				lossb = self.loss_f(f_pred)+self.loss_f(f_pred)+self.loss_f(theta_residual)
				# +self.loss_f(K_residual)
				loss = loss + lossb 
		else:
			_, _, _, f_pred, _, [_, _, _, _, _, _, theta_residual, K_residual] = self.rre_model(residual_data['z'], residual_data['t'], residual_data['flux'])
			loss = self.loss_f(f_pred)+self.loss_f(theta_residual)
			# +self.loss_f(K_residual)
		if log:
			self.loss_log[2].append(loss.numpy())
		return loss

	def loss_boundary_data(self, bound, log = False):
		psi_pred, K_pred, theta_pred, f_pred, flux_pred, [psiz_pred, psit_pred, thetat_pred, Kz_pred, psizz_pred, flux_residual, theta_residual, K_residual] = self.rre_model(bound['z'], bound['t'], bound['flux'])
		_,_,psiweight,fluxweight = self.weight_scheduling()
		if bound['type'] == 'flux':
			loss = self.loss_f(flux_residual)*fluxweight
			if log:
				self.loss_log[7].append(fluxweight.numpy())
		elif bound['type'] == 'psiz':
			loss = self.loss_reduce_mean(psiz_pred, bound['data'])
		elif bound['type'] == 'psi':
			# loss = self.loss_reduce_mean(psi_pred, bound['data'])
			loss = self.loss_reduce_mean(psi_pred, bound['data'])/(self.psi_ub-self.psi_lb)**2*psiweight
			if log:
				self.loss_log[8].append(psiweight.numpy())
		return loss

	# def wrap_trainable_variables(self):
	# 	psi_vars = self.Psi.net.trainable_variables
	# 	K_vars = self.K.net.trainable_variables
	# 	theta_vars = self.Theta.net.trainable_variables
	# 	variables = psi_vars+K_vars+theta_vars
	# 	variables.extend([self.thetas, self.thetar, self.n, self.alpha, self.Ks])
	# 	return variables

	# def get_weights(self):
	# 	w = super().get_weights(convert_to_tensor=False)
	# 	w.extend(self.thetas.numpy())
	# 	w.extend(self.thetar.numpy())
	# 	w.extend(self.n.numpy())
	# 	w.extend(self.alpha.numpy())
	# 	w.extend(self.Ks.numpy())
	# 	w.extend(self.l.numpy())
	# 	return tf.convert_to_tensor(w, dtype=self.dtype)

	# def set_weights(self, w):
	# 	super().set_weights(w)
	# 	self.thetas.assign([w[-5]])
	# 	self.thetar.assign([w[-4]])
	# 	self.n.assign([w[-3]])
	# 	self.alpha.assign([w[-2]])
	# 	self.Ks.assign([w[-1]])
	# 	# self.l.assign([w[-1]])

	# def get_params(self, numpy=False):
	# 	thetas = tf.math.sigmoid(self.thetas)
	# 	thetar = tf.math.sigmoid(self.thetar) 
	# 	n = tf.math.softplus(self.n)+1.0
	# 	alpha = tf.exp(self.alpha)
	# 	Ks = tf.exp(self.Ks)
	# 	# l = tf.math.softplus(self.l)
	# 	if numpy:
	# 		return thetas.numpy()[0], thetar.numpy()[0], n.numpy()[0], alpha.numpy()[0], Ks.numpy()[0]
	# 	return thetas, thetar, n, alpha, Ks

	def plotting(self,epoch):
		if epoch == self.tf_epochs or epoch-self.starting_epoch == 0:
			if self.training_hp['csv_file'] is not None:
				if self.training_hp['csv_file'] == 'sandy_loam_nod.csv':
					[t,z,psi,K,C,theta,flux] = load_csv_data(self.training_hp['csv_file'])

					T = None
					self.test_data_whole = extract_data_test([z,t,theta, K, psi], T = T)
					self.test_data_whole.append(flux_function(self.test_data_whole[1]))

					plot_Ts = [0.1,0.6,1.5,1.6,2.2,2.6]
					self.plot_test_datas = []
					for i in range(len(plot_Ts)):
						T = plot_Ts[i]
						test_data = extract_data_test([z,t,theta, K, psi], T = T)
						test_data.append(flux_function(test_data[1]))
						self.plot_test_datas.append(test_data)

					self.Nt = 251
					self.Nz = int(np.absolute(self.lb.numpy()[0])*10/20)+1

					self.tflux = np.reshape(t,[251,1001])[:,[0]]
					self.zflux = np.zeros(self.tflux.shape)
					self.fluxflux = flux_function(self.tflux)
				elif 'test_plot' in self.training_hp['csv_file']:
					name = self.training_hp['csv_file'].split('.')[0]
					subfix = name.replace('test_plot_data','')
					data = pd.read_csv('./'+self.training_hp['csv_file'] )
					
					Nz = 3
					Nt = int(len(data)/Nz)
					t = data['time'].values[:,None]
					z = data['depth'].values[:,None]
					flux = data['flux'].values[:,None]
					theta = data['theta'].values[:,None]
					T = t[int(Nt/4)]
					test_data = []
					for item in [z,t,flux,theta]:
						Item = np.reshape(item,[Nt,Nz])
						if T is None:
						# Items = Item[int(T/0.012),0:200]
							Items = Item[0:int(Nt/2):8,:]
						else:
							Items = Item[int(Nt/4),:]
						Itemt = np.reshape(Items,[np.prod(Items.shape),1])
						test_data.append(Itemt)
					self.ztest, self.ttest, self.fluxtest, self.thetatest = test_data
					self.zres = np.linspace(-1,-64,64)
					self.tres = T*np.ones(self.zres.shape)
					self.fluxres = self.fluxtest[0]*np.ones(self.zres.shape)

					tbdata = pd.read_csv('./test_plot_tb'+subfix+'.csv')
					t = tbdata['time'].values[:,None]
					z = tbdata['depth'].values[:,None]
					flux = tbdata['flux'].values[:,None]
					flux_test_data = []
					T = None
					# ztb, ttb, fluxtb = extract_data_timewise([z,t,flux],Nt = Nt, Nz = 1, Ns = 0, Ne = int(Nt/2), Ni = 1)
					for item in [z,t,flux]:
						Item = np.reshape(item,[Nt,1])
						if T is None:
						# Items = Item[int(T/0.012),0:200]
							Items = Item[0:int(Nt/2),:]
						else:
							Items = Item[int(Nt/4),:]
						Itemt = np.reshape(Items,[np.prod(Items.shape),1])
						flux_test_data.append(Itemt)
					self.zflux, self.tflux, self.fluxflux = flux_test_data

			self.array_data = []

			for item in self.test_data_whole:
				Item = np.reshape(item,[self.Nt,self.Nz])
				self.array_data.append(Item)

			dt = self.training_hp['dt']
			dz = 2
			self.thetat_dis = (self.array_data[2][1:,:]-self.array_data[2][:-1,:])/dt
			self.psiz_dis = -(self.array_data[4][:,1:]-self.array_data[4][:,:-1])/dz
			self.psit_dis = (self.array_data[4][1:,:]-self.array_data[4][:-1,:])/dt
			self.Kz_dis = -(self.array_data[3][:,1:]-self.array_data[3][:,:-1])/dz
			self.psizz_dis = (self.psiz_dis[:,1:]-self.psiz_dis[:,:-1])/dz
			self.flux_dis = (self.array_data[5][1:,:]-self.array_data[5][:-1,:])/dt
			self.f_dis =  self.thetat_dis[:,1:-1] - self.Kz_dis[1:,1:]*self.psiz_dis[1:,1:]- (self.array_data[3][1:,1:-1])*self.psizz_dis[1:,:] - self.Kz_dis[1:,1:]

		array_data_temp = []

		psi_pred, K_pred, theta_pred, f_residual, _, [psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual, theta_residual, K_residual] = self.rre_model(self.convert_tensor(self.test_data_whole[0]),self.convert_tensor(self.test_data_whole[1]), self.convert_tensor(self.test_data_whole[-1]))
		for item in [f_residual, psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual]:
			Item = np.reshape(item,[self.Nt,self.Nz])
			array_data_temp.append(Item)

		_, _, _, _, flux, [_, _, _, _, _, flux_residual, theta_residual, K_residual] = self.rre_model(self.convert_tensor(self.zflux),self.convert_tensor(self.tflux),self.convert_tensor(self.fluxflux))

		fig1, axs1 = plt.subplots(4, 2)
		# thetat
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

		# f
		axs1[2,1].plot(self.array_data[0][6,1:-1], np.zeros(self.array_data[0][6,1:-1].shape), 'b-')
		axs1[2,1].plot(self.array_data[0][6,1:-1], array_data_temp[0][6,1:-1], 'ro--')
		axs1[2,1].set_title('f vs z')
		axs1[2,1].set(xlabel='z', ylabel='f')
		fig1.suptitle("epoch {0}".format(epoch), fontsize=16)

		# flux
		axs1[3,0].plot(self.tflux, self.fluxflux, 'b-')
		axs1[3,0].plot(self.tflux, flux, 'r--')
		axs1[3,0].set_title('flux vs z')
		axs1[3,0].set(xlabel='z', ylabel='flux')

		# flux residual
		axs1[3,1].plot(self.tflux, np.zeros(self.tflux.shape), 'b-')
		axs1[3,1].plot(self.tflux, flux_residual, 'ro--')
		axs1[3,1].set_title('flux residual vs z')
		axs1[3,1].set(xlabel='z', ylabel='flux f')

		fig1.suptitle("epoch {0}".format(epoch), fontsize=16)

		for i in range(len(self.plot_test_datas)):
			test_data = self.plot_test_datas[i]
			ztest = test_data[0]
			ttest = test_data[1]
			thetatest = test_data[2]
			Ktest = test_data[3]
			psitest = test_data[4]
			fluxtest = test_data[5]
			psi_pred, K_pred, theta_pred, f_residual, flux_pred, [_, _, _, _, _, flux_residual,theta_residual, K_residual] = self.rre_model(self.convert_tensor(ztest),self.convert_tensor(ttest),self.convert_tensor(fluxtest))
			fig, axs = plt.subplots(3, 2)
			axs[0,0].plot(ztest, theta_pred, 'ro--')
			axs[0,0].plot(ztest, thetatest, 'b-')
			axs[0,0].set_title('Theta vs z')
			axs[0,0].set(xlabel='z', ylabel='theta')

			axs[0,1].semilogy(ztest, K_pred, 'ro--')
			axs[0,1].semilogy(ztest, Ktest, 'b-')
			axs[0,1].set_title('K vs z')
			axs[0,1].set(xlabel='z', ylabel='K')

			axs[1,0].plot(ztest, psi_pred, 'ro--')
			axs[1,0].plot(ztest, psitest, 'b-')
			axs[1,0].set_title('Psi vs z')
			axs[1,0].set(xlabel='z', ylabel='psi')

			axs[1,1].plot(psi_pred, theta_pred, 'ro--')
			axs[1,1].plot(psitest, thetatest, 'b-')
			axs[1,1].set_title('Theta vs Psi')
			axs[1,1].set(xlabel='psi', ylabel='theta')

			axs[2,0].plot(ztest, f_residual, 'ro--')
			axs[2,0].plot(ztest, np.zeros(ztest.shape), 'b-')
			axs[2,0].set_title('z vs residual')
			axs[2,0].set(xlabel='z', ylabel='f')

			axs[2,1].plot(ztest, flux_pred, 'ro--')
			axs[2,1].set_title('z vs flux')
			axs[2,1].set(xlabel='z', ylabel='q')
			fig.suptitle("T = {0}".format(ttest[0,0]), fontsize=16)

		filename = "{1}/epoch_{0}.pdf".format(epoch,self.pathname)
		pp = PdfPages(filename)
		fig_nums = plt.get_fignums()
		figs = [plt.figure(n) for n in fig_nums]
		for fig in figs:
			fig.savefig(pp, format='pdf')
		pp.close()
		plt.close('all')

