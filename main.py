import functions

class Synapses:

	tau_r_in = 3.
	tau_d_in = 50.

	tau_r_exc = 1.
	tau_d_exc = 20.

	def __init__(self,N,total_time,dt,length_kernels):

		self.kernels = np.zeros((N,int(length_kernels/dt)),dtype=float)

		for i in range(N):

			if i<(N/2):

				self.kernels[:,i] = functions.alpha_function(self.tau_r_exc,self.tau_d_exc)

			else:
		
				self.kernels[:,i] = -functions.alpha_function(self.tau_r_in,self.tau_d_in)
	
			

class 2LNeuron:
	"""
	LN-LN model. 

	"""

	def __init__(self,dt=0.025,N=10,total_time=60000.,length_kernels=200.)

		self.dt = dt
		self.N = N
		self.total_time = total_time
		self.length_kernels = length_kernels
		
		self.synapses = Synapses(self.N,self.total_time,self.dt,self.length_kernels)

	def run
		
