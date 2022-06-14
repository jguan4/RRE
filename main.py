from RRENetwork import RRENetwork
from RRETestProblem import RRETestProblem


def main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle):

	train_env = RRETestProblem(training_hp['dt'], training_hp['dz'], training_hp['T'], training_hp['Z'],training_hp['noise'], training_hp['name'])
	pathname = train_env.DataFolder
	rrenet = RRENetwork(psistruct, Kstruct, thetastruct, training_hp, pathname)
	if train_toggle == 'train':
		ts, zs, Hs, Ks, Thetas = train_env.get_training_data()
		rrenet.fit(zs,ts,Thetas)
	elif train_toggle == 'test':
		rrenet.load_model()
		test_env = RRETestProblem(test_hp['dt'], test_hp['dz'], test_hp['T'], test_hp['Z'],test_hp['noise'], test_hp['name'])
		t, z, H, K, Theta = test_env.get_testing_data()
		psi_pred, K_pred, theta_pred = rrenet.predict(z,t)
		fig, axs = plt.subplots(2, 2)
		axs[0,0].plot(z, theta_pred, 'r--')
		axs[0,0].plot(z, Theta, 'bs')
		axs[0,0].set_title('Theta vs z')
		axs[0,0].set(xlabel='z', ylabel='theta')

		axs[0,1].semilogy(z, K_pred, 'r--')
		axs[0,1].semilogy(z, K, 'bs')
		axs[0,1].set_title('K vs z')
		axs[0,1].set(xlabel='z', ylabel='K')

		axs[1,0].plot(z, psi_pred, 'r--')
		axs[1,0].plot(z, H, 'bs')
		axs[1,0].set_title('Psi vs z')
		axs[1,0].set(xlabel='z', ylabel='psi')

		axs[1,1].plot(psi_pred, theta_pred, 'r--')
		axs[1,1].plot(H, Theta, 'bs')
		axs[1,1].set_title('Theta vs Psi')
		axs[1,1].set(xlabel='psi', ylabel='theta')

		plt.show()



train_toggle = 'test'
psistruct = {'layers':[2,40,40,40,40,40,40,1]} 
Kstruct = {'layers':[1,40,40,40,1],'toggle':'MNN'} 
thetastruct = {'layers':[1,40,1],'toggle':'MNN'} 
# data = np.genfromtxt(dataname+'.csv',delimiter=',')
training_hp = {'epoch':10000, 'dz': 1, 'dt': 10,'Z':40, 'T':360, 'noise':0, 'name':'Test1'} 
test_hp = {'name':'Test1', 'dz': 0.5, 'dt': 10,'Z':40, 'T':60, 'noise':0}

main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle)