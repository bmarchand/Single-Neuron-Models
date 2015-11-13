import numpy as np
import functions as fun
import random
import matplotlib.pylab as plt

def SpikeGeneration(neuron,control):

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

		if control=='on':

			h = np.histogram(MP_part,bins=1000.,range=[-80.,80.])
			hpost = np.histogram(fun.sigmoid(neuron.non_linearity[g],MP_part),bins=1000.,range=[-80.,80.])

			plt.plot(h[1][:-1],h[0])
			plt.plot(hpost[1][:-1],hpost[0])
			plt.show()

			x = np.arange(-80.,80.,0.1)

			plt.plot(x,fun.sigmoid(neuron.non_linearity[g],x))
			plt.plot(x,x)
			plt.show()


		MP = MP + fun.sigmoid(neuron.non_linearity[g],MP_part)

		

	for t in range(Nsteps):

		lamb = neuron.lambda0*np.exp((MP[t]-neuron.threshold)/neuron.delta_v)

		p = lamb*neuron.dt

		if p>random.random():

			output.append(t*neuron.dt)
	
			MP[t:min(Nsteps,t+neuron.ASP_total_st)] = MP[t:min(Nsteps,t+neuron.ASP_total_st)] - neuron.ASP_size*fun.exp_fun(neuron.ASP_time,neuron.dt,neuron.ASP_total)[:min(neuron.ASP_total_st,Nsteps-t)]

	return output, MP

		

		

		

	
