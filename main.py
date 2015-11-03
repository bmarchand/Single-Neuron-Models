import functions as fun
import numpy as np
import matplotlib.pylab as plt

class Synapses:

	tr_in = 3.
	td_in = 80.

	tr_exc = 1.
	td_exc = 20.
	
	N = 10

	def __init__(self,dt=0.025,len_ker=200.):

		self.dt = dt
		self.len_ker = len_ker
		self.ker = np.zeros((self.N,int(len_ker/dt)),dtype=float)
		

		for i in range(self.N):

			if i<(self.N/2):

				exc_ker = fun.alpha_fun(fun.jitter(self.tr_exc),fun.jitter(self.td_exc),self.dt,self.len_ker)

				self.ker[i,:] = fun.jitter(1.)*exc_ker

			else:

				inh_ker = fun.alpha_fun(fun.jitter(self.tr_in),fun.jitter(self.td_in),self.dt,self.len_ker)
		
				self.ker[i,:] = -fun.jitter(1.)*inh_ker

	def plot(self):

		for i in range(self.N):

			plt.plot(self.ker[i,:])
		
		plt.show()


class TwoLayerNeuron(Synapses):

	def __init__(self,total_time=60000.,rate=10.):
		
		self.synapses = Synapses()
		self.total_time = 60000.
		self.input = []
		self.rate = rate
		self.dt = self.synapses.dt

	def add_input(self):

		for i in range(self.N):

			SpikeTrain = fun.spike_train(fun.jitter(self.rate),self.total_time)

			self.input.append(SpikeTrain)

	def run(self):

		self.output = TwoL

		
		


		
