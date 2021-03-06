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
from matplotlib.backends.backend_pdf import PdfPages

class RRENetwork_flux(RRENetwork):
	def __init__(self, psistruct, Kstruct, thetastruct, training_hp, pathname):
		super().__init__(psistruct, Kstruct, thetastruct, training_hp, pathname)

	def rre_model(self, z_tf, t_tf, flux_tf):

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
		flux_residual = flux-flux_tf

		# f_residual =  theta_t - K_z*psi_z- K*psi_zz - K_z

		# flux = -K*(psi_z+1)

		# return psi, K, theta, f_residual, flux, psi_z
		return psi, K, theta, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual]


	def loss_theta(self, theta_data, log = False):
		_, _, theta_pred, _, _, _ = self.rre_model(theta_data['z'], theta_data['t'], theta_data['flux'])
		loss = self.loss_reduce_mean(theta_pred, theta_data['data'])
		if log:
			self.loss_log[1].append(loss.numpy())
		return loss 

	def loss_residual(self, residual_data, log = False):
		_, _, _, f_pred, _, _ = self.rre_model(residual_data['z'], residual_data['t'], residual_data['flux'])
		loss = self.loss_f(f_pred)
		if log:
			self.loss_log[2].append(loss.numpy())
		return loss

	def loss_boundary_data(self, bound, log = False):
		psi_pred, K_pred, theta_pred, f_pred, flux_pred, [psiz_pred, psit_pred, thetat_pred, Kz_pred, psizz_pred, flux_residual] = self.rre_model(bound['z'], bound['t'], bound['flux'])
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

	def predict(self, z_tf, t_tf, flux_tf):
		z_tf = self.convert_tensor(z_tf)
		t_tf = self.convert_tensor(t_tf)
		flux_tf = self.convert_tensor(flux_tf)
		psi, K, theta, _, _, _ = self.rre_model(z_tf, t_tf, flux_tf)
		return psi.numpy(), K.numpy(), theta.numpy()

	def convert_bound_data(self, bound):
		bound['z'] = self.convert_tensor(bound['z'])
		bound['t'] = self.convert_tensor(bound['t'])
		bound['flux'] = self.convert_tensor(bound['flux'])
		bound['data'] = self.convert_tensor(bound['data'])
		return bound

	def convert_residual_data(self, residual):
		residual['z'] = self.convert_tensor(residual['z'])
		residual['t'] = self.convert_tensor(residual['t'])
		residual['flux'] = self.convert_tensor(residual['flux'])
		return residual

	def plotting(self,epoch):
		if epoch == 0 or epoch == self.tf_epochs:
			if self.training_hp['csv_file'] is not None:
				if self.training_hp['csv_file'] == 'sandy_loam_nod.csv':

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
							Items = Item[:,:int(np.absolute(self.lb.numpy()[0])*10):20]
						else:
							Items = Item[int(T/0.012),:int(np.absolute(self.lb.numpy()[0])*10):20]
						Itemt = np.reshape(Items,[np.prod(Items.shape),1])
						test_data.append(Itemt)
					self.ztest_whole, self.ttest_whole, self.thetatest_whole, self.Ktest_whole, self.psitest_whole = test_data
					self.flux_whole = self.flux_function(self.ttest_whole)
					test_data = []
					T = 0.6
					for item in [z,t,theta, K, psi]:
						Item = np.reshape(item,[251,1001])
						if T is None:
						# Items = Item[int(T/0.012),0:200]
							Items = Item[:,:int(np.absolute(self.lb.numpy()[0])*10):20]
						else:
							Items = Item[int(T/0.012),:int(np.absolute(self.lb.numpy()[0])*10):20]
						Itemt = np.reshape(Items,[np.prod(Items.shape),1])
						test_data.append(Itemt)
					self.ztest, self.ttest, self.thetatest, self.Ktest, self.psitest = test_data
					self.fluxtest = self.flux_function(self.ttest)
					self.Nt = 251
					self.Nz = int(np.absolute(self.lb.numpy()[0])*10/20)
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

			else:
				test_env = RRETestProblem(self.training_hp['dt'], self.training_hp['dz'], self.training_hp['T'], self.training_hp['Z'],self.training_hp['noise'], self.training_hp['name'],'')
				self.ttest_whole, self.ztest_whole, self.psitest_whole, self.Ktest_whole, self.thetatest_whole = test_env.get_training_data()

				test_env = RRETestProblem(self.training_hp['dt'], self.training_hp['dz'], 70, self.training_hp['Z'],self.training_hp['noise'], self.training_hp['name'],'')
				self.ttest, self.ztest, self.psitest, self.Ktest, self.thetatest = test_env.get_testing_data()
				self.Nt = int(self.training_hp['T']/self.training_hp['dt'])
				self.Nz = int(self.training_hp['Z']/self.training_hp['dz'])+1

			# self.array_data = []

			# for item in [self.ztest_whole, self.ttest_whole, self.thetatest_whole, self.Ktest_whole, self.psitest_whole, self.flux_whole]:
			# 	Item = np.reshape(item,[self.Nt,self.Nz])
			# 	self.array_data.append(Item)

			# self.thetat_dis = (self.array_data[2][1:,:]-self.array_data[2][:-1,:])/self.training_hp['dt']
			# self.psiz_dis = -(self.array_data[4][:,1:]-self.array_data[4][:,:-1])/self.training_hp['dz']
			# self.psit_dis = (self.array_data[4][1:,:]-self.array_data[4][:-1,:])/self.training_hp['dt']
			# self.Kz_dis = -(self.array_data[3][:,1:]-self.array_data[3][:,:-1])/self.training_hp['dz']
			# self.psizz_dis = (self.psiz_dis[:,1:]-self.psiz_dis[:,:-1])/self.training_hp['dz']
			# self.flux_dis = (self.array_data[5][1:,:]-self.array_data[5][:-1,:])/self.training_hp['dt']
			# self.f_dis =  self.thetat_dis[:,1:-1] - self.Kz_dis[1:,1:]*self.psiz_dis[1:,1:]- (self.array_data[3][1:,1:-1])*self.psizz_dis[1:,:] - self.Kz_dis[1:,1:]

		# array_data_temp = []

		# psi_pred, K_pred, theta_pred, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual] = self.rre_model(self.convert_tensor(self.ztest_whole),self.convert_tensor(self.ttest_whole), self.convert_tensor(self.flux_whole))
		# for item in [f_residual, psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual]:
			# Item = np.reshape(item,[self.Nt,self.Nz])
			# array_data_temp.append(Item)

		# psi_pred, K_pred, theta_pred = self.predict(self.ztest,self.ttest,self.fluxtest)
		_, _, theta_pred, _, _, _ = self.rre_model(self.convert_tensor(self.ztest),self.convert_tensor(self.ttest),self.convert_tensor(self.fluxtest))
		_, _, _, f_residual, _, _ = self.rre_model(self.convert_tensor(self.zres),self.convert_tensor(self.tres),self.convert_tensor(self.fluxres))
		_, _, _, _, _, [_, _, _, _, _, flux_residual] = self.rre_model(self.convert_tensor(self.zflux),self.convert_tensor(self.tflux),self.convert_tensor(self.fluxflux))
		# fig1, axs1 = plt.subplots(4, 2)
		#thetat
		# axs1[0,0].plot(self.array_data[0][6,:], self.thetat_dis[6,:], 'b-')
		# axs1[0,0].plot(self.array_data[0][6,:], array_data_temp[3][6,:], 'ro--')
		# axs1[0,0].set_title('Theta_t vs z')
		# axs1[0,0].set(xlabel='z', ylabel='theta_t')

		# #psiz
		# axs1[0,1].plot(self.array_data[0][6,1:], self.psiz_dis[6,:], 'b-')
		# axs1[0,1].plot(self.array_data[0][6,1:], array_data_temp[1][6,1:], 'ro--')
		# axs1[0,1].set_title('psi_z vs z')
		# axs1[0,1].set(xlabel='z', ylabel='psi_z')

		# #psit
		# axs1[1,0].plot(self.array_data[0][6,:], self.psit_dis[6,:], 'b-')
		# axs1[1,0].plot(self.array_data[0][6,:], array_data_temp[2][6,:], 'ro--')
		# axs1[1,0].set_title('psi_t vs z')
		# axs1[1,0].set(xlabel='z', ylabel='psi_t')

		# #Kz
		# axs1[1,1].plot(self.array_data[0][6,1:], self.Kz_dis[6,:], 'b-')
		# axs1[1,1].plot(self.array_data[0][6,1:], array_data_temp[4][6,1:], 'ro--')
		# axs1[1,1].set_title('K_z vs z')
		# axs1[1,1].set(xlabel='z', ylabel='K_z')

		# #psizz
		# axs1[2,0].plot(self.array_data[0][6,1:-1], self.psizz_dis[6,:], 'b-')
		# axs1[2,0].plot(self.array_data[0][6,1:-1], array_data_temp[5][6,1:-1], 'ro--')
		# axs1[2,0].set_title('psi_zz vs z')
		# axs1[2,0].set(xlabel='z', ylabel='psi_zz')

		#f
		# axs1[2,1].plot(self.array_data[0][6,1:-1], np.zeros(self.array_data[0][6,1:-1].shape), 'b-')
		# # axs1[2,1].plot(self.array_data[0][6,1:-1], np.zeros(self.f_dis[6,:].s, 'ro--')
		# axs1[2,1].plot(self.array_data[0][6,1:-1], array_data_temp[0][6,1:-1], 'ro--')
		# axs1[2,1].set_title('f vs z')
		# axs1[2,1].set(xlabel='z', ylabel='f')
		# fig1.suptitle("epoch {0}".format(epoch), fontsize=16)

		# axs1[3,0].plot(self.array_data[0][6,1:-1], np.zeros(self.array_data[0][6,1:-1].shape), 'b-')
		# # axs1[3,0].plot(self.array_data[0][6,1:-1], np.zeros(self.f_dis[6,:].s, 'ro--')
		# axs1[3,0].plot(self.array_data[0][6,1:-1], array_data_temp[6][6,1:-1], 'ro--')
		# axs1[3,0].set_title('flux_residual vs z')
		# axs1[3,0].set(xlabel='z', ylabel='flux f')
		# fig1.suptitle("epoch {0}".format(epoch), fontsize=16)
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

		axs[0,1].plot(self.zres, np.zeros(self.zres.shape), 'b-')
		# axs1[2,1].plot(self.array_data[0][6,1:-1], np.zeros(self.f_dis[6,:].s, 'ro--')
		axs[0,1].plot(self.zres, f_residual, 'ro--')
		axs[0,1].set_title('f vs z')
		axs[0,1].set(xlabel='z', ylabel='f')

		axs[1,0].plot(self.tflux, np.zeros(self.tflux.shape), 'b-')
		# axs1[3,0].plot(self.array_data[0][6,1:-1], np.zeros(self.f_dis[6,:].s, 'ro--')
		axs[1,0].plot(self.tflux, flux_residual, 'ro--')
		axs[1,0].set_title('flux_residual vs z')
		axs[1,0].set(xlabel='t', ylabel='flux f')
		fig.suptitle("epoch {0}, T = {1}".format(epoch,self.ttest[0,0]), fontsize=16)

		# axs[0,1].semilogy(self.ztest, K_pred, 'ro--')
		# axs[0,1].semilogy(self.ztest, self.Ktest, 'b-')
		# axs[0,1].set_title('K vs z')
		# axs[0,1].set(xlabel='z', ylabel='K')

		# axs[1,0].plot(self.ztest, psi_pred, 'ro--')
		# axs[1,0].plot(self.ztest, self.psitest, 'b-')
		# axs[1,0].set_title('Psi vs z')
		# axs[1,0].set(xlabel='z', ylabel='psi')

		# axs[1,1].plot(psi_pred, theta_pred, 'ro--')
		# axs[1,1].plot(self.psitest, self.thetatest, 'b-')
		# axs[1,1].set_title('Theta vs Psi')
		# axs[1,1].set(xlabel='psi', ylabel='theta')
		# fig.suptitle("T = {0}".format(self.ttest[0,0]), fontsize=16)

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

	def flux_function(self, t, toggle = 'Bandai1'):
		flux = np.zeros(t.shape)
		if toggle == 'Bandai1':
			flux[np.argwhere(np.logical_and(t>=0,t<0.25))] = -10
			flux[np.argwhere(np.logical_and(t>=0.5,t<1.0))] = 0.3
			flux[np.argwhere(np.logical_and(t>=1.5,t<2.0))] = 0.3
			flux[np.argwhere(np.logical_and(t>=2.0,t<2.25))] = -10
			flux[np.argwhere(np.logical_and(t>=2.5,t<=3.0))] = 0.3
		return flux
