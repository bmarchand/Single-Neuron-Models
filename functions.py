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

def exp_fun(tau,dt,L,control='off'):

	x = np.arange(int(L/dt))

	tau_steps = tau/dt

	y = np.exp(-x/tau_steps)

	if control=='on':
		plt.plot(x*dt,y)
		plt.show()

	return y

def jitter(r,per=0.1):

	return r*(1+per*2*(random.random()-0.5))

def spike_train(neuron):

	train = []

	while np.sum(train)<neuron.total_time:

		tmp_t = np.random.exponential(scale=1000./jitter(neuron.in_rate))

		tmp_t = neuron.dt*int(tmp_t/neuron.dt)

		train.append(tmp_t)

	train = np.cumsum(train[:-1])

	train = list(train)

	return train
		
def sigmoid(l,x):

	y = 0.5*(l[0]/(l[1]+np.exp(-l[2]*x)) - l[0]/(l[1]+np.exp(l[2]*x)))

	return y


def CosineBasis(K,a,c,T,dt):

    F = np.zeros((T/dt,K),dtype='float')

    I = np.arange(T/dt)*dt

    for k in range(K):

        lb = math.exp((k/2.-1)*math.pi/a) - c

        ub = math.exp((k/2.+1)*math.pi/a) - c

        F[(I>=lb)&(I<ub),k] = 0.5*(1+np.cos(a*np.log(I[(I>=lb)&(I<ub)]+c)-k*0.5*math.pi))

    return F

def NaturalSpline(knots,bnds):
	
	dv = 0.001*(bnds[1]-bnds[0])

	F = np.zeros((1000.,len(knots))

	v = np.arange(bnds[0],bnds[1],dv)

	F[:,0] = 1.

	F[:,1] = v

	for i in range(2,len(knots)):

		tmp1 = (v-knots[i])**3
		tmp1[tmp1<0.] = 0.

		tmp2 = (v-knots[-1])**3
		tmp2[tmp2<0.] = 0.

		tmp3 = (v-knots[i-1])**3
		tmp3[tmp3<0.] = 0.		

		dk = (tmp1 - tmp2)*(1./(knots[-1]-knots[i]))

		dkm1 = (tmp3 - tmp2)*(1./(knots[-1]-knots[i-1]))

		F[:,i] = dk - dkm1

	return F

def SplineParamsforId(knots,bnds):

	F = NaturalSpline(knots,bnds)

	dv = 0.001*(bnds[1]-bnds[0])

	v = np.arange(bnds[0],bnds[1],dv)

	params = np.dot(np.linalg.inv(np.dot(F.transpose(),F)),np.dot(F.transpose(),v))

	return params
	





