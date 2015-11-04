import functions as fun
import numpy as np
import matplotlib.pylab as plt
import mechanisms as mech
import random

class Synapses:

	tr_in = 3.
	td_in = 80.

	tr_exc = 1.
	td_exc = 20.
	
	N = 12

	def __init__(self,size,dt=0.025,len_ker=200.):

		self.dt = dt
		self.len_ker = len_ker
		self.len_ker_st = int(len_ker/dt)
		self.ker = np.zeros((self.N,int(len_ker/dt)),dtype=float)

		l = [0,1,2,6,7,8,3,4,5,9,10,11]
		 
		for i in range(self.N):

			if i<(self.N/2):

				exc_ker = fun.alpha_fun(fun.jitter(self.tr_exc),fun.jitter(self.td_exc),self.dt,self.len_ker)

				self.ker[l[i],:] = fun.jitter(size)*exc_ker

			else:

				inh_ker = fun.alpha_fun(fun.jitter(self.tr_in),fun.jitter(self.td_in),self.dt,self.len_ker)
		
				self.ker[l[i],:] = -fun.jitter(size)*inh_ker

	def plot(self):

		for i in range(self.N):

			plt.plot(self.ker[i,:])
		
		plt.show()

class SpikingMechanism:

	dt = 0.025
	compartments = 2
	non_linearity = [[80,1.,0.05],[100.,1.,0.04]]
	threshold = 30.
	PSP_size = 20. # mV: Roughly the std of the membrane potential in one compartment before NL
	ASP_size = 20.
	ASP_time = 200.
	ASP_total = 1000.
	ASP_total_st = ASP_total/dt
	delta_v = 5.
	lambda0 = 0.1

class RunParameters:

	total_time = 100000.
	in_rate = 20.
	dt = 0.025

class TwoLayerNeuron(Synapses,SpikingMechanism,RunParameters):

	def __init__(self):

		self.synapses = Synapses(self.PSP_size)
		self.add_input()

	def add_input(self):
	
		self.input = []

		for i in range(self.N):

			SpikeTrain = fun.spike_train(fun.jitter(self.in_rate),self.total_time,self.dt)

			self.input.append(SpikeTrain)

	def run(self):

		self.output,self.membrane_potential = mech.SpikeGeneration(self)
	
		self.output_rate = len(self.output)/(0.001*self.total_time)

	def plot(self):

		plt.plot(np.arange(int(self.total_time/self.dt)),self.membrane_potential)
		plt.show()
		

		
		


		
