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

				exc_ker = fun.alpha_fun(fun.jitter(self.tr_exc),fun.jitter(self.td_exc))

				self.ker[i,:] = fun.jitter(1.)*exc_ker

			else:

				inh_ker = fun.alpha_fun(fun.jitter(self.tr_in),fun.jitter(self.td_in))
		
				self.ker[i,:] = -fun.jitter(1.)*inh_ker

	def plot(self):

		for i in range(self.N):

			plt.plot(self.ker[i,:])
		
		plt.show()

class TwoLayerNeuron(Synapses):

	def __init__(self,):
		
		self.synapses = Synapses()

	
