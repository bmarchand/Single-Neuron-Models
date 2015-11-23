import numpy as np
import functions as fun
import random
import matplotlib.pylab as plt
import copy

def SpikeGeneration(neuron,control):

	Nsteps = int(neuron.total_time/neuron.dt)

	MP = np.zeros((Nsteps,),dtype='float')

	inp_tmp = copy.copy(neuron.input)

	output = []

	for g in range(neuron.compartments):

		MP_part = np.zeros((Nsteps,),dtype='float')

		nsyn = int(neuron.N/neuron.compartments)

		for cnt in range(nsyn):

			in_syn = inp_tmp.pop()

			for t in in_syn:

				t = int(t/neuron.dt)

				len_ker = neuron.synapses.len_ker_st
				bndup = min(t+len_ker,Nsteps)
				bndupk = min(len_ker,Nsteps-t)
				size = neuron.PSP_size
				kerns = neuron.synapses.ker
				indik = int(g*nsyn+cnt)

				MP_part[t:bndup] = MP_part[t:bndup] + size*kerns[indik,:bndupk]

		if control=='on':

			h = np.histogram(MP_part,bins=1000.,range=[-80.,80.])

			MP_afterNL = fun.sigmoid(neuron.non_linearity[g],MP_part)
			hpost = np.histogram(MP_afterNL,bins=1000.,range=[-80.,80.])

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
			bndup = min(Nsteps,t+neuron.ASP_total_st)
			bndupk = min(neuron.ASP_total_st,Nsteps-t)
			size = neuron.ASP_size
			expfun = fun.exp_fun(neuron.ASP_time,neuron.dt,neuron.ASP_total)

			MP[t:bndup] = MP[t:bndup] - size*expfun[:bndupk]

	return output, MP
	
