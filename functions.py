import math
import numpy as np
import matplotlib.pylab as plt
import random

def alpha_fun(tau_r,tau_d,dt=0.025,L=200.,control='off'):

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
