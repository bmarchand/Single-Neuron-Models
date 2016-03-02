import numpy as np
import functions as fun #defined in functions.py
import random
import matplotlib.pylab as plt
import copy
import os

import diff_calc as diff

def SpikeGeneration(inp,neuron,control,string):#neuron contains input spike-trains and a sp mech 

	if string=='training':

		T = neuron.total_time

	elif string=='test':

		T = neuron.total_time_test

	Nsteps = int(T/neuron.dt) #total number of timesteps

	MP = np.zeros((Nsteps,),dtype='float') #membrane potential that the model uses.
                                  
	inp_tmp = copy.copy(inp) #if no copy input is emptied (we use pop)

	output = [] #where output spike trains are stored.

	for g in range(neuron.Ng): #Ng = groups with their own NL

		MP_part = np.zeros((Nsteps,),dtype='float') #memb.pot. before NL within comp. 

		nsyn = int(neuron.N/neuron.Ng) #number of synapse in a group.

		for cnt in range(nsyn): #loop over the synapses in the compartment/group

			in_syn = inp_tmp.pop() #inp_tmp is a list of list of input spike trains.S

			for t in in_syn: # input spikes (in ms)

				t = int(t/neuron.dt) # conversion ms -> time-step number.

				len_ker = neuron.synapses.len_ker_st #[time-steps] 
				bndup = min(t+len_ker,Nsteps) 
				bndupk = min(len_ker,Nsteps-t) #because could go beyons maximal size.
				size = neuron.PSP_size #~25 in our case.
				kerns = neuron.synapses.ker # ~1 of size.
				indik = int(g*nsyn+cnt) # kerns is for the entire neuron.

				MP_part[t:bndup] = MP_part[t:bndup] + size*kerns[indik,:bndupk]

# Why not fftconvolve ? -> because would imply create a huge array of zeros and ones.
# and this is not a slow part of the code anyway. + for loop is "sparse".

		if control=='on':
		
			plt.plot(MP_part)
			print g
			plt.plot(fun.sigmoid(neuron.non_linearity[g],MP_part))
			plt.show()
		
			h = np.histogram(MP_part,bins=1000.,range=[-2.,2.])
			h_after = np.histogram(fun.sigmoid(neuron.non_linearity[g],MP_part),bins=1000.,range=[-2.,2.])

			plt.plot(h[1][:-1],h[0])
			plt.plot(h_after[1][:-1],h_after[0])
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

	return output, MP, MP_part

def run_model(inp,model):

	Nsteps = int(model.total_time_test/model.dt) # total number of time-steps.

	MP = np.zeros(Nsteps)
	Nneurons = int(model.N/model.Ng) # number of neurons in compartment.
	Nbnl = np.shape(model.basisNL[0])[0] # number of basis functions for NL.

	MP12 = model.sub_memb_pot_test

	for g in range(model.Ng): #loop over compartments. 

		Mp_g = MP12[g,:] 

		F = model.basisNL[g]
		NL = np.dot(model.paramNL[g*Nbnl:(g+1)*Nbnl],F)

		dv = (model.bnds[g][1] - model.bnds[g][0])*0.00001

		Mp_g = (Mp_g-model.bnds[g][0])/dv
		Mp_g = np.floor(Mp_g)
		Mp_g = Mp_g.astype('int')

		Mp_g[Mp_g>99999] = 99999
		Mp_g[Mp_g<0] = 0

		Mp_g = NL[Mp_g] #NL is an array with mV.

		MP = MP + Mp_g

	Nbasp = model.basisASP.shape[0]

	ASP = np.dot(model.paramKer[(-Nbasp-1):-1],model.basisASP)

	out_model_test = []

	for t in range(Nsteps):

		lamb = np.exp(MP[t] - model.paramKer[-1])

		p = lamb*model.dt

		if random.random() < p:

			out_model_test = out_model_test + [t*model.dt]

			bndup = min(Nsteps,t + ASP.size)
			bndupk = min(ASP.size,Nsteps-t)
			
			MP[t:bndup] = MP[t:bndup] - ASP[:bndupk] # can't convolve

	return  out_model_test

	
