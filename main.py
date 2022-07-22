from RRENetwork import RRENetwork
from RRETestProblem import RRETestProblem
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from utils import *

def load_data(training_hp, csv_file = None):
	if csv_file is not None:
		if csv_file == 'sandy_loam_nod.csv':
			EnvFolder = "./RRE_Bandai_{1}domain_{0}_weight_halftimetheta_checkpoints".format(training_hp['scheduleing_toggle'],np.absolute(training_hp['lb'][0]))
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
			# zf1 = np.linspace(-25,0,251)
			# zf2 = np.linspace(-100,-26,75)
			# zff = np.hstack((zf1, zf2))
			# tff = np.linspace(0,3,251)
			# Z,T = np.meshgrid(zff,tff)
			# zf = Z.flatten()
			# tf = T.flatten()
			# zf, tf = extract_data_time_space([z,t], Nt = 251, Nz = 1001, Ns = 0, Ne = 1001, Ni = 3, Nst = 0, Net = 251, Nit = 1)
			zf, tf = extract_data([z,t], Ne = np.absolute(training_hp['lb'][0])*10, Ni = 10)
			ztb, ttb, fluxtb = extract_top_boundary([z,t,flux])
			zbb, tbb, psibb = extract_bottom_boundary([z,t,psi], Nb = np.absolute(training_hp['lb'][0])*10)

			flux_inputs = []
			for item in [tt,tf,ttb,tbb]:
				flux_inputs.append(flux_function(item))

			fluxt, fluxf, fluxtb, fluxbb = flux_inputs
			
			theta_data = {'z':zt, 't':tt, 'data':thetat}
			residual_data = {'z':zf, 't':tf}
			boundary_data = {'top':{'z':ztb, 't':ttb, 'data':fluxtb, 'type':'flux'}, 'bottom':{'z':zbb, 't':tbb, 'data':psibb, 'type':'psi'}}
	
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
		data = pd.read_csv('./'+csv_file)
		t = data['time'].values[:,None]
		z = data['depth'].values[:,None]
		psi = data['head'].values[:,None]
		K = data['K'].values[:,None]
		C = data['C'].values[:,None]
		theta = data['theta'].values[:,None]
		flux = data['flux'].values[:,None]
		ztest_whole, ttest_whole, thetatest_whole, Ktest_whole, psitest_whole = extract_data_test(t,z,theta, K, psi)

		ztest, ttest, thetatest, Ktest, psitest = extract_data_test(t, z, theta, K, psi, T = 0.6)
	else: 
		test_env = RRETestProblem(test_hp['dt'], test_hp['dz'], test_hp['T'], test_hp['Z'],test_hp['noise'], test_hp['name'],'')
		ttest_whole, ztest_whole, psitest_whole, Ktest_whole, thetatest_whole = test_env.get_training_data()

		test_env = RRETestProblem(test_hp['dt'], test_hp['dz'], 70, test_hp['Z'],test_hp['noise'], test_hp['name'],'')
		ttest, ztest, psitest, Ktest, thetatest = test_env.get_testing_data()

	return ztest_whole, ttest_whole, thetatest_whole, Ktest_whole, psitest_whole, ttest, ztest, psitest, Ktest, thetatest


def main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle, csv_file):
	theta_data, residual_data, boundary_data, pathname = load_data(training_hp, csv_file = csv_file)
	rrenet = RRENetwork(psistruct, Kstruct, thetastruct, training_hp, pathname)
	if train_toggle == 'train':
		rrenet.fit(theta_data, residual_data,boundary_data)

	elif train_toggle == 'retrain':
		rrenet.load_model()
		rrenet.fit(theta_data, residual_data,boundary_data)

	elif train_toggle == 'test':
		rrenet.load_model()

		ztest_whole, ttest_whole, thetatest_whole, Ktest_whole, psitest_whole, ttest, ztest, psitest, Ktest, thetatest = test_data(test_hp, csv_file)

		# psi_pred, K_pred, theta_pred = rrenet.predict(ztest_whole, ttest_whole)
		psi_pred, K_pred, theta_pred, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz] = rrenet.rre_model(rrenet.convert_tensor(ztest_whole),rrenet.convert_tensor(ttest_whole))
		Nt = int(test_hp['T']/test_hp['dt'])+1
		Nz = int(test_hp['Z']/test_hp['dz'])
		array_data = []

		for item in [ztest_whole, ttest_whole, thetatest_whole, Ktest_whole, psitest_whole, f_residual, psi_z, psi_t, theta_t, K_z, psi_zz]:
			Item = np.reshape(item,[Nt,Nz])
			array_data.append(Item)

		thetat_dis = (array_data[2][1:,:]-array_data[2][:-1,:])/test_hp['dt']
		psiz_dis = -(array_data[4][:,1:]-array_data[4][:,:-1])/test_hp['dz']
		psit_dis = (array_data[4][1:,:]-array_data[4][:-1,:])/test_hp['dt']
		Kz_dis = -(array_data[3][:,1:]-array_data[3][:,:-1])/test_hp['dz']
		psizz_dis = (psiz_dis[:,1:]-psiz_dis[:,:-1])/test_hp['dz']
		f_dis =  thetat_dis[:,1:-1] - Kz_dis[1:,1:]*psiz_dis[1:,1:]- (array_data[3][1:,1:-1])*psizz_dis[1:,:] - Kz_dis[1:,1:]

		fig1, axs1 = plt.subplots(3, 2)
		#thetat
		axs1[0,0].plot(array_data[0][6,:], thetat_dis[6,:], 'b-')
		axs1[0,0].plot(array_data[0][6,:], array_data[8][6,:], 'ro--')
		axs1[0,0].set_title('Theta_t vs z')
		axs1[0,0].set(xlabel='z', ylabel='theta_t')

		#psiz
		axs1[0,1].plot(array_data[0][6,1:], psiz_dis[6,:], 'b-')
		axs1[0,1].plot(array_data[0][6,1:], array_data[6][6,1:], 'ro--')
		axs1[0,1].set_title('psi_z vs z')
		axs1[0,1].set(xlabel='z', ylabel='psi_z')

		#psit
		axs1[1,0].plot(array_data[0][6,:], psit_dis[6,:], 'b-')
		axs1[1,0].plot(array_data[0][6,:], array_data[7][6,:], 'ro--')
		axs1[1,0].set_title('psi_t vs z')
		axs1[1,0].set(xlabel='z', ylabel='psi_t')

		#Kz
		axs1[1,1].plot(array_data[0][6,1:], Kz_dis[6,:], 'b-')
		axs1[1,1].plot(array_data[0][6,1:], array_data[9][6,1:], 'ro--')
		axs1[1,1].set_title('K_z vs z')
		axs1[1,1].set(xlabel='z', ylabel='K_z')

		#psizz
		axs1[2,0].plot(array_data[0][6,1:-1], psizz_dis[6,:], 'b-')
		axs1[2,0].plot(array_data[0][6,1:-1], array_data[10][6,1:-1], 'ro--')
		axs1[2,0].set_title('psi_zz vs z')
		axs1[2,0].set(xlabel='z', ylabel='psi_zz')

		#f
		axs1[2,1].plot(array_data[0][6,1:-1], f_dis[6,:], 'b-')
		axs1[2,1].plot(array_data[0][6,1:-1], array_data[5][6,1:-1], 'ro--')
		axs1[2,1].set_title('f vs z')
		axs1[2,1].set(xlabel='z', ylabel='f')
		# plt.show()

		order_str = ['theta','K','psi']
		for (ostr,data, pred) in zip(order_str,[thetatest_whole, Ktest_whole, psitest_whole], [theta_pred,K_pred,psi_pred]):
			err = relative_error(data,pred)
			print("For {0}, relative error is {1}.\n".format(ostr,err))

		psi_pred, K_pred, theta_pred = rrenet.predict(ztest,ttest)

		fig, axs = plt.subplots(2, 2)
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

		plt.show()


train_toggle = 'train'
starting_epoch = 0
name = 'Test1'
# csv_file = None 
csv_file = "sandy_loam_nod.csv" 
if csv_file is not None:
	# lb = [-20,0]
	# lb = [-50,0]
	lb = [-100,0]
	ub = [0,3]
else:
	if name == 'Test1':
		lb = [0,0]
		ub = [40,360]
psistruct = {'layers':[2,40,40,40,40,40,40,1]} 
Kstruct = {'layers':[1,40,1],'toggle':'MNN'} 
thetastruct = {'layers':[1,40,1],'toggle':'MNN'} 
# data = np.genfromtxt(dataname+'.csv',delimiter=',')
lbfgs_options={'disp': None, 'maxcor': 50, 'ftol': 2.220446049250313e-16, 'gtol': 1e-09, 'maxfun': 50000, 'maxiter': 50000, 'maxls': 50, 'iprint':1}
adam_options = {'epoch':8950}
total_epoch = lbfgs_options['maxiter'] + adam_options['epoch']
# 'dz': 1, 'dt': 10,'Z':40, 'T':360
training_hp = {'dz': 0.1, 'dt': 0.012,'Z':100, 'T':3, 'noise':0,'lb':lb,'ub':ub, 'name':name,'lbfgs_options':lbfgs_options, 'adam_options':adam_options, 'norm':'_norm1', 'weights': [1, 1, 1e3, 1], 'csv_file':csv_file, 'psi_lb':-1000,'psi_ub':-12.225, 'starting_epoch': starting_epoch, 'total_epoch':total_epoch, 'scheduleing_toggle':'constant'}

test_hp = {'name':'Test1', 'dz': 0.1, 'dt': .012,'Z':100, 'T':3, 'noise':0}

main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle, csv_file)