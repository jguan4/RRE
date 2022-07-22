from RRENetwork_flux import RRENetwork_flux
from RRENetwork_flux_ide import RRENetwork_flux_ide
from RRETestProblem import RRETestProblem
import pandas as pd
import numpy as np
import os
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from utils import *

def load_data(training_hp, csv_file = None):
	if csv_file is not None:
		if csv_file == 'sandy_loam_nod.csv':
			tags = '_ACTUALhalftimetheta_flin_Knn'
			if training_hp['network_toggle'] == 'RRENetwork_flux':
				EnvFolder = "./RRE_Bandai_{1}domain_{0}_weight_flux{2}_checkpoints".format(training_hp['scheduleing_toggle'],np.absolute(training_hp['lb'][0]),tags)
			elif training_hp['network_toggle'] == 'RRENetwork_flux_ide':
				EnvFolder = "./RRE_Bandai_{1}domain_{0}_weight_flux_ide{2}_checkpoints".format(training_hp['scheduleing_toggle'],np.absolute(training_hp['lb'][0]),tags)
			# EnvFolder = './RRE_Bandai_100domain_20lb_checkpoints'
			Nt = int(training_hp['T']/training_hp['dt'])
			Nz = int(training_hp['Z']/training_hp['dz'])+1
			weights = training_hp['weights']
			add_tag = "_f{0}_flux{2}_theta{1}".format(weights[1], weights[2], weights[3])
			if weights[0] == 0:
				weight_tag = ''
			else:
				weight_tag = "_weight{0}".format(weights[0])
			DataFolder = EnvFolder+"/Nt{0}_Nz{1}_noise{2}{3}{4}{5}".format(251,10,training_hp['noise'],training_hp['norm'], weight_tag,add_tag)

			if not os.path.exists(EnvFolder):
				os.makedirs(EnvFolder)
			if not os.path.exists(DataFolder):
				os.makedirs(DataFolder)
			pathname = DataFolder

			data = pd.read_csv('./'+csv_file)
			t = data['time'].values[:,None]
			z = data['depth'].values[:,None]
			psi = data['head'].values[:,None]
			K = data['K'].values[:,None]
			C = data['C'].values[:,None]
			theta = data['theta'].values[:,None]
			flux = data['flux'].values[:,None]
			zt, tt, thetat = extract_data_time_space([z,t,theta], Nt = 251, Nz = 1001, Ns = 1, Ne = 200, Ni = 20, Nst = 0, Net = 125, Nit = 1)
			# zt, tt, thetat = extract_data([z,t,theta])
			zf1 = np.linspace(-25,0,26) #densef1
			# zf2 = np.linspace(-100,-26,186) #densef1
			zf2 = np.linspace(-100,-26,75) #not dense
			zff = np.hstack((zf2, zf1))
			tff = np.linspace(0,3,251)
			Z,T = np.meshgrid(zff,tff)
			zf = np.reshape(Z,[np.prod(Z.shape),1])
			tf = np.reshape(T,[np.prod(T.shape),1])
			# zf, tf = extract_data_time_space([z,t], Nt = 251, Nz = 1001, Ns = 0, Ne = 1001, Ni = 10, Nst = 0, Net = 251, Nit = 1)
			# zf, tf = extract_data([z,t], Ne = np.absolute(training_hp['lb'][0])*10, Ni = 10)
			ztb, ttb, fluxtb = extract_top_boundary([z,t,flux])
			zbb, tbb, psibb = extract_bottom_boundary([z,t,psi], Nb = np.absolute(training_hp['lb'][0])*10)

			flux_inputs = []
			for item in [tt,tf,ttb,tbb]:
				flux_inputs.append(flux_function(item))

			fluxt, fluxf, fluxtb, fluxbb = flux_inputs
			
			theta_data = {'z':zt, 't':tt, 'flux':fluxt, 'data':thetat}
			residual_data = {'z':zf, 't':tf, 'flux':fluxf}
			boundary_data = {'top':{'z':ztb, 't':ttb, 'flux':fluxtb, 'data':fluxtb, 'type':'flux'}, 'bottom':{'z':zbb, 't':tbb, 'flux':fluxbb, 'data':psibb, 'type':'psi'}}

		elif 'test_plot' in csv_file:
		# elif csv_file == 'test_plot_data.csv':
			name = csv_file.split('.')[0]
			subfix = name.replace('test_plot_data','')
			data = pd.read_csv('./'+csv_file)
			EnvFolder = "./RRE_{2}_{1}domain_halftime_{0}_weight_fluxc_checkpoints".format(training_hp['scheduleing_toggle'],np.absolute(training_hp['lb'][0]),name)
			# EnvFolder = './RRE_Bandai_100domain_20lb_checkpoints'
			Nz = 3
			Nt = int(len(data)/Nz)
			weights = training_hp['weights']
			add_tag = "_f{0}_flux{2}_theta{1}".format(weights[1], weights[2], weights[3])
			if weights[0] == 0:
				weight_tag = ''
			else:
				weight_tag = "_weight{0}".format(weights[0])
			DataFolder = EnvFolder+"/Nt{0}_Nz{1}_noise{2}{3}{4}{5}_1".format(Nt,Nz,training_hp['noise'],training_hp['norm'], weight_tag,add_tag)

			if not os.path.exists(EnvFolder):
				os.makedirs(EnvFolder)
			if not os.path.exists(DataFolder):
				os.makedirs(DataFolder)
			pathname = DataFolder

			t = data['time'].values[:,None]
			z = data['depth'].values[:,None]
			flux = data['flux'].values[:,None]
			theta = data['theta'].values[:,None]
			zt, tt, fluxt, thetat = extract_data_timewise([z,t,flux,theta],Nt = Nt, Nz = Nz, Ns = 0, Ne = int(Nt/2), Ni = 1)

			tbdata = pd.read_csv('./test_plot_tb'+subfix+'.csv')
			t = tbdata['time'].values[:,None]
			z = tbdata['depth'].values[:,None]
			flux = tbdata['flux'].values[:,None]
			ztb, ttb, fluxtb = extract_data_timewise([z,t,flux],Nt = Nt, Nz = 1, Ns = 0, Ne = int(Nt/2), Ni = 1)

			bbdata = pd.read_csv('./test_plot_bb'+subfix+'.csv')
			t = bbdata['time'].values[:,None]
			z = bbdata['depth'].values[:,None]
			flux = bbdata['flux'].values[:,None]
			psi = bbdata['psi'].values[:,None]
			zbb, tbb, fluxbb, psibb = extract_data_timewise([z,t,flux,psi],Nt = Nt, Nz = 1, Ns = 0, Ne = int(Nt/2), Ni = 1)

			zs = np.linspace(-65,0,27)
			zs = zs[1:-1]
			ts = tt[::30] 
			fluxs = fluxt[::30] 
			Z,T = np.meshgrid(zs,ts)
			Fluxs = np.tile(np.reshape(fluxs,[len(fluxs),1]),(1,len(zs)))
			zf = Z.flatten()
			tf = T.flatten()
			fluxf = Fluxs.flatten()

			theta_data = {'z':zt, 't':tt, 'flux':fluxt, 'data':thetat}
			residual_data = {'z':zf, 't':tf, 'flux':fluxf}
			boundary_data = {'top':{'z':ztb, 't':ttb, 'flux':fluxtb, 'data':fluxtb, 'type':'flux'}, 'bottom':{'z':zbb, 't':tbb, 'flux':fluxbb, 'data':psibb, 'type':'psi'}}
	else:
		weights = training_hp['weights']
		add_tag = "_f{0}_flux{2}_theta{1}".format(weights[1], weights[2], weights[3])
		# add_tag = "_f{0}".format(weights[1])
		if weights[0] == 0:
			weight_tag = ''
		else:
			weight_tag = "_weight{0}".format(weights[0])
		tag = training_hp['norm']+weight_tag+add_tag
		train_env = RRETestProblem(training_hp['dt'], training_hp['dz'], training_hp['T'], training_hp['Z'],training_hp['noise'], training_hp['name'], tag)
		pathname = train_env.DataFolder
		ts, zs, Hs, Ks, Thetas = train_env.get_training_data()
		zt = zs 
		tt = ts 
		thetat = Thetas
		zbb, tbb, psibb = extract_bottom_boundary(ts,zs,Hs,Nt = train_env.env.Nt, Nz = train_env.env.Nz)
		ztb, ttb, psitb = extract_top_boundary(ts,zs,Hs,Nt = train_env.env.Nt, Nz = train_env.env.Nz)
		theta_data = {'z':zt, 't':tt, 'data':thetat}
		residual_data = {'z':zt, 't':tt}
		boundary_data = {'top':{'z':ztb, 't':ttb, 'data':psitb, 'type':'psi'}, 'bottom':{'z':zbb, 't':tbb, 'data':psibb, 'type':'psi'}}
	return theta_data, residual_data, boundary_data, pathname

def test_data(test_hp, csv_file = None):
	if csv_file is not None:
		if csv_file == 'sandy_loam_nod.csv':
			data = pd.read_csv('./'+csv_file)
			t = data['time'].values[:,None]
			z = data['depth'].values[:,None]
			psi = data['head'].values[:,None]
			K = data['K'].values[:,None]
			C = data['C'].values[:,None]
			theta = data['theta'].values[:,None]
			flux = data['flux'].values[:,None]
			ztest_whole, ttest_whole, thetatest_whole, Ktest_whole, psitest_whole = extract_data_test([z,t,theta, K, psi])

			flux_whole = flux_function(ttest_whole)

			ztest, ttest, thetatest, Ktest, psitest = extract_data_test([z,t, theta, K, psi], T = 0.6)
			fluxtest = flux_function(ttest)
		elif csv_file == 'test_plot_data.csv':
			data = pd.read_csv('./'+csv_file)
			ttest_whole = data['time'].values[:,None]
			ztest_whole = data['depth'].values[:,None]
			flux_whole = data['flux'].values[:,None]
			thetatest_whole = data['theta'].values[:,None]
			Ktest_whole = None 
			psitest_whole = None 
			ttest = None
			ztest = None
			fluxtest = None 
			psitest = None
			Ktest = None
			thetatest = None

	else: 
		test_env = RRETestProblem(test_hp['dt'], test_hp['dz'], test_hp['T'], test_hp['Z'],test_hp['noise'], test_hp['name'],'')
		ttest_whole, ztest_whole, psitest_whole, Ktest_whole, thetatest_whole = test_env.get_training_data()

		test_env = RRETestProblem(test_hp['dt'], test_hp['dz'], 70, test_hp['Z'],test_hp['noise'], test_hp['name'],'')
		ttest, ztest, psitest, Ktest, thetatest = test_env.get_testing_data()

	return ztest_whole, ttest_whole, flux_whole, thetatest_whole, Ktest_whole, psitest_whole, ttest, ztest, fluxtest, psitest, Ktest, thetatest


def main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle, csv_file):
	theta_data, residual_data, boundary_data, pathname = load_data(training_hp, csv_file = csv_file)
	print(theta_data['t'])
	input()
	if training_hp['network_toggle'] == 'RRENetwork_flux':
		rrenet = RRENetwork_flux(psistruct, Kstruct, thetastruct, training_hp, pathname)
	elif training_hp['network_toggle'] == 'RRENetwork_flux_ide':
		rrenet = RRENetwork_flux_ide(psistruct, Kstruct, thetastruct, training_hp, pathname)
	if train_toggle == 'train':
		rrenet.fit(theta_data, residual_data,boundary_data)

	elif train_toggle == 'retrain':
		rrenet.load_model()
		rrenet.fit(theta_data, residual_data,boundary_data)

	elif train_toggle == 'test':
		rrenet.load_model()
		# Nt = 19006
		# Nz = 3
		Nt = 251
		Nz = 200


		ztest_whole, ttest_whole, flux_whole, thetatest_whole, Ktest_whole, psitest_whole, ttest, ztest, fluxtest, psitest, Ktest, thetatest = test_data(test_hp, csv_file)

		zt, tt, fluxt, thetat, Kt, psit = extract_data_time_space([ztest_whole,ttest_whole,flux_whole,thetatest_whole,Ktest_whole,psitest_whole],Ns = 0, Ne = 200, Ni = 1)

		# psi_pred, K_pred, theta_pred = rrenet.predict(ztest_whole, ttest_whole)
		psi_pred, K_pred, theta_pred, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual] = rrenet.rre_model(rrenet.convert_tensor(zt),rrenet.convert_tensor(tt), rrenet.convert_tensor(fluxt))
		diff =(theta_pred-thetat)
		# diff =np.absolute(theta_pred-thetat)
		diff_mat = np.reshape(diff, [int(Nt), Nz])
		fluxt_mat = np.reshape(fluxt, [int(Nt), Nz])
		fluxp_mat = np.reshape(flux, [int(Nt), Nz])
		tt_mat = np.reshape(tt, [int(Nt), Nz])
		fig,ax = plt.subplots(2,1)
		for i in range(Nz):
			ax[0].plot(tt_mat[:,0],diff_mat[:,i])
		ax[1].plot(tt_mat[:,0],fluxt_mat[:,0])
		ax[1].plot(tt_mat[:,0],fluxp_mat[:,0])
		plt.show()
		
		# psi_pred, K_pred, theta_pred, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual] = rrenet.rre_model(rrenet.convert_tensor(ztest_whole),rrenet.convert_tensor(ttest_whole), rrenet.convert_tensor(flux_whole))
		# Nt = int(test_hp['T']/test_hp['dt'])+1
		# Nz = int(test_hp['Z']/test_hp['dz'])
		# array_data = []

		# for item in [ztest_whole, ttest_whole, thetatest_whole, Ktest_whole, psitest_whole, f_residual, psi_z, psi_t, theta_t, K_z, psi_zz]:
		# 	Item = np.reshape(item,[Nt,Nz])
		# 	array_data.append(Item)

		# thetat_dis = (array_data[2][1:,:]-array_data[2][:-1,:])/test_hp['dt']
		# psiz_dis = -(array_data[4][:,1:]-array_data[4][:,:-1])/test_hp['dz']
		# psit_dis = (array_data[4][1:,:]-array_data[4][:-1,:])/test_hp['dt']
		# Kz_dis = -(array_data[3][:,1:]-array_data[3][:,:-1])/test_hp['dz']
		# psizz_dis = (psiz_dis[:,1:]-psiz_dis[:,:-1])/test_hp['dz']
		# f_dis =  thetat_dis[:,1:-1] - Kz_dis[1:,1:]*psiz_dis[1:,1:]- (array_data[3][1:,1:-1])*psizz_dis[1:,:] - Kz_dis[1:,1:]

		# fig1, axs1 = plt.subplots(3, 2)
		# #thetat
		# axs1[0,0].plot(array_data[0][6,:], thetat_dis[6,:], 'b-')
		# axs1[0,0].plot(array_data[0][6,:], array_data[8][6,:], 'ro--')
		# axs1[0,0].set_title('Theta_t vs z')
		# axs1[0,0].set(xlabel='z', ylabel='theta_t')

		# #psiz
		# axs1[0,1].plot(array_data[0][6,1:], psiz_dis[6,:], 'b-')
		# axs1[0,1].plot(array_data[0][6,1:], array_data[6][6,1:], 'ro--')
		# axs1[0,1].set_title('psi_z vs z')
		# axs1[0,1].set(xlabel='z', ylabel='psi_z')

		# #psit
		# axs1[1,0].plot(array_data[0][6,:], psit_dis[6,:], 'b-')
		# axs1[1,0].plot(array_data[0][6,:], array_data[7][6,:], 'ro--')
		# axs1[1,0].set_title('psi_t vs z')
		# axs1[1,0].set(xlabel='z', ylabel='psi_t')

		# #Kz
		# axs1[1,1].plot(array_data[0][6,1:], Kz_dis[6,:], 'b-')
		# axs1[1,1].plot(array_data[0][6,1:], array_data[9][6,1:], 'ro--')
		# axs1[1,1].set_title('K_z vs z')
		# axs1[1,1].set(xlabel='z', ylabel='K_z')

		# #psizz
		# axs1[2,0].plot(array_data[0][6,1:-1], psizz_dis[6,:], 'b-')
		# axs1[2,0].plot(array_data[0][6,1:-1], array_data[10][6,1:-1], 'ro--')
		# axs1[2,0].set_title('psi_zz vs z')
		# axs1[2,0].set(xlabel='z', ylabel='psi_zz')

		# #f
		# axs1[2,1].plot(array_data[0][6,1:-1], f_dis[6,:], 'b-')
		# axs1[2,1].plot(array_data[0][6,1:-1], array_data[5][6,1:-1], 'ro--')
		# axs1[2,1].set_title('f vs z')
		# axs1[2,1].set(xlabel='z', ylabel='f')
		# # plt.show()

		order_str = ['theta','K','psi']
		for (ostr,data, pred) in zip(order_str,[thetat, Kt, psit], [theta_pred,K_pred,psi_pred]):
			err = relative_error(data,pred)
			print("For {0}, relative error is {1}.\n".format(ostr,err))

		# order_str = ['theta']
		# for (ostr,data, pred) in zip(order_str,[thetatest_whole], [theta_pred]):
		# 	err = relative_error(data,pred)
		# 	print("For {0}, relative error is {1}.\n".format(ostr,err))

		# psi_pred, K_pred, theta_pred = rrenet.predict(ztest,ttest)

		# fig, axs = plt.subplots(2, 2)
		# axs[0,0].plot(ztest, theta_pred, 'ro--')
		# axs[0,0].plot(ztest, thetatest, 'b-')
		# axs[0,0].set_title('Theta vs z')
		# axs[0,0].set(xlabel='z', ylabel='theta')

		# axs[0,1].semilogy(ztest, K_pred, 'ro--')
		# axs[0,1].semilogy(ztest, Ktest, 'b-')
		# axs[0,1].set_title('K vs z')
		# axs[0,1].set(xlabel='z', ylabel='K')

		# axs[1,0].plot(ztest, psi_pred, 'ro--')
		# axs[1,0].plot(ztest, psitest, 'b-')
		# axs[1,0].set_title('Psi vs z')
		# axs[1,0].set(xlabel='z', ylabel='psi')

		# axs[1,1].plot(psi_pred, theta_pred, 'ro--')
		# axs[1,1].plot(psitest, thetatest, 'b-')
		# axs[1,1].set_title('Theta vs Psi')
		# axs[1,1].set(xlabel='psi', ylabel='theta')

		# plt.show()


train_toggle = 'train'
starting_epoch = 0
name = 'Test1'
# csv_file = None 
csv_file = "sandy_loam_nod.csv" 
if csv_file is not None:
	if csv_file == 'sandy_loam_nod.csv':
		# lb = [-20,0]
		lb = [-100,0,0]
		ub = [0,3,1]
	elif 'test_plot' in csv_file:
		lb = [-65,0,0]
		# lb = [-65,0,-10]
		ub = [0,198,1]
		# ub = [0,198,0]
else:
	if name == 'Test1':
		lb = [0,0]
		ub = [40,360]
psistruct = {'layers':[3,40,40,40,40,40,40,1]} 
Kstruct = {'layers':[1,40,40,40,1],'toggle':'NN'} 
thetastruct = {'layers':[1,40,1],'toggle':'MNN'} 
# data = np.genfromtxt(dataname+'.csv',delimiter=',')
lbfgs_options={'disp': None, 'maxcor': 50, 'ftol': 2.220446049250313e-16, 'gtol': 1e-09, 'maxfun': 50000, 'maxiter': 50000, 'maxls': 50, 'iprint':1}
adam_options = {'epoch':10000}
total_epoch = lbfgs_options['maxiter'] + adam_options['epoch']
# 'dz': 1, 'dt': 10,'Z':40, 'T':360
training_hp = {'dz': 15, 'dt': 0.0104,'Z':65, 'T':198, 'noise':0,'lb':lb,'ub':ub, 'name':name,'lbfgs_options':lbfgs_options, 'adam_options':adam_options, 'norm':'_norm', 'weights': [1, 1e-3, 2e3, 1], 'csv_file':csv_file, 'psi_lb':-1000,'psi_ub':-12.225, 'starting_epoch': starting_epoch, 'total_epoch':total_epoch, 'scheduleing_toggle':'linear', 'network_toggle':'RRENetwork_flux'}
# training_hp = {'dz': 15, 'dt': 0.0104,'Z':65, 'T':198, 'noise':0,'lb':lb,'ub':ub, 'name':name,'lbfgs_options':lbfgs_options, 'adam_options':adam_options, 'norm':'_norm', 'weights': [1, 1, 5e2, 1], 'csv_file':csv_file, 'psi_lb':-270,'psi_ub':0, 'starting_epoch': starting_epoch, 'total_epoch':total_epoch, 'scheduleing_toggle':'constant'}

test_hp = {'name':'Test1', 'dz': 0.1, 'dt': .012,'Z':100, 'T':3, 'noise':0}

main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle, csv_file)