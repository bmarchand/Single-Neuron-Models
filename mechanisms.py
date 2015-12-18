import numpy as np
import functions as fun #defined in functions.py
import random
import matplotlib.pylab as plt
import copy
import os

def SpikeGeneration(neuron,control):#neuron contains input spike-trains and a sp mech 

	Nsteps = int(neuron.total_time/neuron.dt) #total number of timesteps

	MP = np.zeros((Nsteps,),dtype='float') #membrane potential that the model uses.
                                  
	inp_tmp = copy.copy(neuron.input) #if no copy input is emptied (we use pop)

	output = [] #where output spike trains are stored.

	for g in range(neuron.compartments): #compartments = groups with their own NL

		MP_part = np.zeros((Nsteps,),dtype='float') #memb.pot. before NL within comp. 

		nsyn = int(neuron.N/neuron.compartments) #number of synapse in a group.

		for cnt in range(nsyn): #loop over the synapses in the compartment/group

			in_syn = inp_tmp.pop() #inp_tmp is a list of list of input spike trains.S

			for t in in_syn: # input spikes (in ms)

				t = int(t/neuron.dt) # conversion ms -> time-step number.

				len_ker = neuron.synapses.len_ker_st #[time-steps] 
				bndup = min(t+len_ker,Nsteps) 
				bndupk = min(len_ker,Nsteps-t) #because could go beyons maximal size.
				size = neuron.PSP_size #~25 in our case.
				kerns = neuron.synapses.ker # ~1 of size.
				indik = int(g*nsyn+cnt) # kerns is for the entire neurons.

				MP_part[t:bndup] = MP_part[t:bndup] + size*kerns[indik,:bndupk]

# Why not fftconvolve ? -> because would imply create a huge array of zeros and ones.
# and this is not a slow part of the code anyway. + for loop is "sparse".

		plt.plot(MP_part)
		plt.plot(fun.sigmoid(neuron.non_linearity[g],MP_part))
		plt.show()
		
		h = np.histogram(MP_part,bins=1000.,range=[-80.,80.])
		h_after = np.histogram(fun.sigmoid(neuron.non_linearity[g],MP_part),bins=1000.,range=[-80.,80.])

		plt.plot(h[1][:-1],h[0])
		plt.plot(h_after[1][:-1],h_after[0])
		plt.show()

		x = np.arange(-80.,80.,0.1)

		plt.plot(x,fun.sigmoid(neuron.non_linearity[g],x))
		plt.plot(x,x)
		plt.show()

		MP = MP + fun.sigmoid(neuron.non_linearity[g],MP_part) 

	for t in range(Nsteps): #spike generation itself.

		lamb = neuron.lambda0*np.exp((MP[t]-neuron.threshold)/neuron.delta_v)

		p = lamb*neuron.dt #first-order of (1-exp(-lamb*dt))

		if p>random.random():

			output.append(t*neuron.dt) #convert back to [ms]
			bndup = min(Nsteps,t+neuron.ASP_total_st)
			bndupk = min(neuron.ASP_total_st,Nsteps-t)
			size = neuron.ASP_size # ~30
			expfun = fun.exp_fun(neuron.ASP_time,neuron.dt,neuron.ASP_total) # ~1

			MP[t:bndup] = MP[t:bndup] - size*expfun[:bndupk] # can't convolve here.

	return output, MP
	
