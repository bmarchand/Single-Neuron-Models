import numpy as np
from scipy import signal
import copy
import matplotlib.pylab as plt
from scipy import weave
from scipy.weave import converters
import os

def convolve(spike_train,basis,state):
	"""Each spike train has to be convoluted with all the basis functions. 
	spike_train is a list of lists of spike times. basis is a numpy array 
	of dimensions (number_of_basis_functions,length_ker_in_ms/dt). 
	state is included in case we need dt or something else. This is a general
	purpose convolution function. I'm doing here what I didn't do for spike
	generation. That's not very coherent. """

	Nsteps = int(state.total_time/state.dt)  

	Nb = copy.copy(np.shape(basis)[0]) #nb of basis func. copy because np.atleast_2d.

	ST = np.zeros((len(spike_train),Nsteps),dtype='float') #huge array with 0s and 1s.
										   # len(spike_train) = number of spiketrains.
	lent = len(spike_train)

	for i in range(lent): #number of spite-trains

		indices = np.around(np.array(spike_train[i])*(1./state.dt))#convrt to timestep
		indices = indices + [1.] #effect of a spike arrives right after spike
		indices = indices.astype('int') 
		indices = indices[indices<Nsteps] #+1 could move timestep beyond boundary.

		ST[i,indices] = 1.

	X = np.zeros((lent*Nb,Nsteps),dtype='float')

	for i in range(Nb):

		vec_i = np.atleast_2d(basis[i,:]) #need 2d because broadcast to ST shape.
		X[lent*i:lent*(i+1),:] = signal.fftconvolve(ST,vec_i)[:,:Nsteps] 
	
	return X

def convolve_for(spike_train,basis,state):

	Nsteps = int(state.total_time/state.dt)
	lent = len(spike_train)
	Nb = copy.copy(np.shape(basis)[0])
	lb = np.shape(basis)[1]

	X = np.zeros((Nb*lent,Nsteps),dtype='float')

	for i in range(lent):

		st = np.around(np.array(spike_train[i])*(1/state.dt))
		st = st.astype('int')
		
		for b in range(Nb):
			for t in st:
					
				bndlo = t + 1
				bndup = min(t+lb+1,Nsteps)

				X[i*b,bndlo:bndup] = X[i*b,bndlo:bndup] + basis[b,:(bndup-bndlo)]

	return X

def convolve_c(spike_train,basis,state):

	Nsteps = int(state.total_time/state.dt)

	Nb = copy.copy(np.shape(basis)[0])
	
	lb = np.shape(basis)[1]

	lent = len(spike_train)

	lengths = np.array(map(len,spike_train))

	st = map(np.array,spike_train)
	
	def multiply(a):
		return a*(1./state.dt)

	st = map(multiply,st)
	st = map(np.around,st)
	st = map(np.uint,st)	

	X = np.zeros((Nb*lent,Nsteps),dtype='double')

	code = """
	#include <math.h>

	
	int N = lent;
	int nb = Nb;
	int Lb = lb;
	int ii,jj,kk,ll;

	for (ii = 0; ii < N; ii++) {
		for (jj = 0; jj<Nb; jj++){
			for (kk = 0; kk < lengths[ii]; kk++){

				int bnddo;
				int bndup;
				int t;
				t = st[ii,kk];
				bndup = t + Lb + 1;
				bnddo = t + 1;
				if (N<bndup+1){
					 bndup = N;
							  } 

				for (ll = bnddo; ll < bndup; ll++){
					int b = ii*jj;
					int indi;
					indi = bndup - bnddo;
					double r = basis[jj,indi];
					X[b,ll] = X[b,ll] + r; 												   }			
												 }
						     	  }
						        }
	"""	
		
	variables = ['Nb','lent','Nsteps','basis','st','lengths','lb','X']		

	weave.inline(code,variables)

	return X
