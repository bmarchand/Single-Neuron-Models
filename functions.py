import math
import numpy as np
import matplotlib.pylab as plt
import random

def alpha_fun(tau_r,tau_d,dt,L,control='off'):

	x = np.arange(int(L/dt))

	tau_r_steps = tau_r/dt
	tau_d_steps = tau_d/dt

	y = np.exp(-x/tau_r_steps)-np.exp(-x/tau_d_steps)

	y = y/np.sum(y*dt)

	if control=='on':
		plt.plot(x*dt,y)
		plt.show()

	return y

def jitter(r,per=0.1):

	return r*(1+per*2*(random.random()-0.5))

def spike_train(rate,total_time):

	train = []

	while np.sum(train)<total_time:

		train.append(np.random.exponential(scale=1000./rate))

	train = np.cumsum(train[:-1])

	train = list(train)

	return train
		

	
