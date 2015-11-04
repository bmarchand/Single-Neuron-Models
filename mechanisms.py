import numpy as np
import functions as fun
import random

def SpikeGeneration(neuron):

	Nsteps = int(neuron.total_time/neuron.dt)

	MP = np.zeros((Nsteps,),dtype='float')

	output = []

	for g in range(neuron.compartments):

		MP_part = np.zeros((Nsteps,),dtype='float')

		nsyn = int(neuron.N/neuron.compartments)

		for cnt in range(nsyn):

			in_syn = neuron.input.pop()

			for t in in_syn:

				t = int(t/neuron.dt)

				MP_part[t:min(t+neuron.synapses.len_ker_st,Nsteps)] = MP_part[t:min(t+neuron.synapses.len_ker_st,Nsteps)] + neuron.PSP_size*neuron.synapses.ker[int(g*nsyn+cnt),:min(neuron.synapses.len_ker_st,Nsteps-t)]

		MP = MP + fun.sigmoid(neuron.non_linearity[g],MP_part)

	for t in range(Nsteps):

		lamb = neuron.lambda0*np.exp((MP[t]-neuron.threshold)/neuron.delta_v)

		p = lamb*neuron.dt

		if p>random.random():

			output.append(t*neuron.dt)
	
			MP[t:min(Nsteps,t+neuron.ASP_total_st)] = MP[t:min(Nsteps,t+neuron.ASP_total_st)] - fun.exp_fun(neuron.ASP_time,neuron.dt,neuron.ASP_total)[:min(neuron.ASP_total_st,Nsteps-t)]

	return output, MP

		

		

		

	
