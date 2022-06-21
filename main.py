from RRENetwork import RRENetwork
from RRETestProblem import RRETestProblem
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def extract_data(t,z,theta, Nt = 251, Nz = 1001, Ns = 10, Ne = 200, Ni = 20):
	train_data = []
	for item in [z,t,theta]:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[:,Ns:Ne:Ni]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_top_boundary(t,z,vec, Nt = 251, Nz = 1001):
	train_data = []
	for item in [z,t,vec]:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[:,[0]]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_bottom_boundary(t,z,vec,Nt = 251, Nz = 1001, Nb = None):
	train_data = []
	for item in [z,t,vec]:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[:,[-1]] if Nb is None else Item[:,[Nb]]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_data_test(t,z,theta, K, psi, T = None):
	test_data = []
	for item in [z,t,theta, K, psi]:
		Item = np.reshape(item,[251,1001])
		if T is None:
		# Items = Item[int(T/0.012),0:200]
			Items = Item[:,:200]
		else:
			Items = Item[int(T/0.012),0:200]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		test_data.append(Itemt)
	return test_data

def relative_error(data, pred):
	# data = np.reshape(data,[251,200])
	# pred = np.reshape(pred,[251,200])
	error = np.sum((data-pred)**2,axis = None)/np.sum(data**2,axis = None)
	return error

def psi_func(theta):
	thetas = 0.41
	thetar = 0.065
	n = 1.89
	m = 1-1/n
	alpha = 0.075
	Psi = (((thetas-thetar)/(theta-thetar))**(1/m)-1)**(1/n)/(-alpha)
	return Psi

def load_data(training_hp, csv_file = None):
	if csv_file is not None:
		EnvFolder = './RRE_Bandai_checkpoints'
		Nt = int(training_hp['T']/training_hp['dt'])
		Nz = int(training_hp['Z']/training_hp['dz'])+1
		if training_hp['weight'] == 0:
			weight_tag = ''
		else:
			weight_tag = "_weight{0}".format(training_hp['weight'])
		DataFolder = EnvFolder+"/Nt{0}_Nz{1}_noise{2}{3}{4}".format(251,10,training_hp['noise'],training_hp['norm'], weight_tag)

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
		zt, tt, thetat = extract_data(t,z,theta)
		ztb, ttb, fluxtb = extract_top_boundary(t,z,flux)
		zbb, tbb, psibb = extract_bottom_boundary(t,z,psi, Nb = 200)

		boundary_data = {'top':{'z':ztb, 't':ttb, 'data':fluxtb, 'type':'flux'}, 'bottom':{'z':zbb, 't':tbb, 'data':psibb, 'type':'psi'}}
	else:
		if training_hp['weight'] == 0:
			weight_tag = ''
		else:
			weight_tag = "_weight{0}".format(training_hp['weight'])
		tag = training_hp['norm']+weight_tag
		train_env = RRETestProblem(training_hp['dt'], training_hp['dz'], training_hp['T'], training_hp['Z'],training_hp['noise'], training_hp['name'], tag)
		pathname = train_env.DataFolder
		ts, zs, Hs, Ks, Thetas = train_env.get_training_data()
		zt = zs 
		tt = ts 
		thetat = Thetas
		zbb, tbb, psibb = extract_bottom_boundary(ts,zs,Hs,Nt = train_env.env.Nt, Nz = train_env.env.Nz)
		ztb, ttb, psitb = extract_top_boundary(ts,zs,Hs,Nt = train_env.env.Nt, Nz = train_env.env.Nz)
		boundary_data = {'top':{'z':ztb, 't':ttb, 'data':psitb, 'type':'psi'}, 'bottom':{'z':zbb, 't':tbb, 'data':psibb, 'type':'psi'}}
	return zt, tt, thetat, boundary_data, pathname

def test_data(test_hp, csv_file = None):
	if csv_file is not None:
		data = pd.read_csv(f"./sandy_loam_nod.csv")
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
	zt, tt, thetat, boundary_data, pathname = load_data(training_hp, csv_file = csv_file)
	rrenet = RRENetwork(psistruct, Kstruct, thetastruct, training_hp, pathname)
	if train_toggle == 'train':
		rrenet.fit(zt,tt,thetat,boundary_data)

	elif train_toggle == 'retrain':
		rrenet.load_model()
		rrenet.fit(zt,tt,thetat,boundary_data)

	elif train_toggle == 'test':
		rrenet.load_model()

		ztest_whole, ttest_whole, thetatest_whole, Ktest_whole, psitest_whole, ttest, ztest, psitest, Ktest, thetatest = test_data(test_hp, csv_file)
		psi_pred, K_pred, theta_pred = rrenet.predict(ztest_whole, ttest_whole)

		order_str = ['theta','K','psi']
		for (ostr,data, pred) in zip(order_str,[thetatest_whole, Ktest_whole, psitest_whole], [theta_pred,K_pred,psi_pred]):
			err = relative_error(data,pred)
			print("For {0}, relative error is {1}.\n".format(ostr,err))

		# psi_pred, K_pred, theta_pred = rrenet.predict(ztest,ttest)
		psi_pred, K_pred, theta_pred, f_residual, flux, psi_z = rrenet.rre_model(rrenet.convert_tensor(ztest),rrenet.convert_tensor(ttest))
		print(f_residual)
		input()

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


train_toggle = 'test'
name = 'Test1'
csv_file = None 
# csv_file = "sandy_loam_nod.csv" 
if csv_file is not None:
	lb = [-20,0]
	ub = [0,3]
else:
	if name == 'Test1':
		lb = [0,0]
		ub = [40,360]
psistruct = {'layers':[2,40,40,40,40,40,40,1]} 
Kstruct = {'layers':[1,40,40,40,1],'toggle':'MNN'} 
thetastruct = {'layers':[1,40,1],'toggle':'MNN'} 
# data = np.genfromtxt(dataname+'.csv',delimiter=',')
lbfgs_options={'disp': None, 'maxcor': 50, 'ftol': 2.220446049250313e-16, 'maxfun': 10000, 'maxiter': 10000, 'maxls': 50, 'iprint':1}
training_hp = {'dz': 1, 'dt': 10,'Z':40, 'T':360, 'noise':0,'lb':lb,'ub':ub, 'name':name,'lbfgs_options':lbfgs_options, 'adam_options':{'epoch':10000}, 'norm':'_norm1', 'weight': 0.05}
test_hp = {'name':'Test1', 'dz': 0.5, 'dt': 10,'Z':40, 'T':360, 'noise':0}

main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle, csv_file)