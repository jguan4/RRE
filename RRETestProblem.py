import numpy as np
import os
from RRE_test1 import RRE_test1


class RRETestProblem(object):
	def __init__(self, dt, dz, T, Z, noise, name):
		self.name = name
		self.noise = noise
		if self.name == 'Test1':
			self.env = RRE_test1(dt, dz, T, Z)
		self.define_folder_names()

	def get_training_data(self):
		trainfile_exist, testfile_exist = self.generate_folder()
		if trainfile_exist:
			ts, zs, Hs, Ks, Thetas = self.load_data(self.TrainDataName)
		else:
			print("calling {0}...\n".format(self.name))
			ts, zs, Hs, Thetas, Ks = self.env.get_data(train_toggle = 'training')
			# standard normal distibuiton with 0 mean and stndard error of the noise value
			noise_theta = self.noise*np.random.randn(Thetas.shape[0], Thetas.shape[1]) 
			Thetas = Thetas + noise_theta
			self.save_data(ts, zs, Hs, Ks, Thetas, self.TrainDataName)
		return ts, zs, Hs, Ks, Thetas

	def get_testing_data(self):
		# trainfile_exist, testfile_exist = self.generate_folder()
		
		print("calling {0}...\n".format(self.name))
		t, z, H, Theta, K = self.env.get_data(train_toggle = 'testing')
		# standard normal distibuiton with 0 mean and stndard error of the noise value
		# noise_theta = self.noise*np.random.randn(Thetas.shape[0], Thetas.shape[1]) 
		# Thetas = Thetas + noise_theta
		# self.save_data(ts, zs, Hs, Ks, Thetas)
		return t, z, H, K, Theta

	def define_folder_names(self):
		self.EnvFolder = './RRE_'+self.name+'_checkpoints'
		self.DataFolder = self.EnvFolder+"/Nt{0}_Nz{1}_noise{2}".format(self.env.Nt,self.env.Nz,self.noise)
		self.TrainDataName = self.DataFolder+'/training_data.npz'
		self.TestDataName = self.DataFolder+'/testing_data.npz'

	def generate_folder(self):
		if not os.path.exists(self.EnvFolder):
			os.makedirs(self.EnvFolder)
		if not os.path.exists(self.DataFolder):
			os.makedirs(self.DataFolder)
		trainfile_exist = os.path.exists(self.TrainDataName)
		testfile_exist = os.path.exists(self.TestDataName)
		return trainfile_exist, testfile_exist

	def save_data(self, ts, zs, Hs, Ks, Thetas, name):
		np.savez(name, ts = ts, zs= zs, Hs = Hs, Ks = Ks, Thetas = Thetas)

	def load_data(self, name):
		file = np.load(name)
		return file['ts'], file['zs'], file['Hs'], file['Ks'], file['Thetas']