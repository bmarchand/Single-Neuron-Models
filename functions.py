import math
import numpy as np
import matplotlib.pylab as plt

def alpha_function(tau_r,tau_d,dt=0.025,L=100.,control='off'):

	x = np.arange(int(L/dt))

	tau_r_steps = tau_r/dt
	tau_d_steps = tau_d/dt

	y = np.exp(-x/tau_r_steps)-np.exp(-x/tau_d_steps)

	y = y/np.sum(y*dt)

	if control=='on':
		plt.plot(x*dt,y)
		plt.show()

	return y
