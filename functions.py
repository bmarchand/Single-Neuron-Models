import math
import numpy as np
import matplotlib.pylab as plt
import random

def alpha_fun(tau_r,tau_d,dt,L,control='off'): #alpha functions for kernels.
 
	x = np.arange(int(L/dt)) #abcsiss vector in timesteps.

	tau_r_steps = tau_r/dt # in time steps.
	tau_d_steps = tau_d/dt

	y = np.exp(-x/tau_r_steps)-np.exp(-x/tau_d_steps)

	y = y/np.sum(y*dt)

	if control=='on':
		plt.plot(x*dt,y) #x*dt is in ms.
		plt.show()

	return y

def exp_fun(tau,dt,L,control='off'): #exponential function for ASP.

	x = np.arange(int(L/dt)) 

	tau_steps = tau/dt

	y = np.exp(-x/tau_steps)

	if control=='on':
		plt.plot(x*dt,y)
		plt.show()

	return y

def jitter(r,per=0.1): #add 10% of noise to r.

	return r*(1+per*2*(random.random()-0.5))

def spike_train(neuron): #generate Poisson spike train.

	train = []

	while np.sum(train)<neuron.total_time:

		tmp_t = np.random.exponential(scale=1000./jitter(neuron.in_rate)) #ISI

		tmp_t = neuron.dt*int(tmp_t/neuron.dt)

		train.append(tmp_t)

	train = np.cumsum(train[:-1]) 

	train = list(train)

	return train
		
def sigmoid(l,x):

	y = l[0] + l[1]/( l[2]+np.exp(-l[3]*x) )

	return y

def DerSigmoid(l,x):

	y = l[1]*l[3]*np.exp(-l[3]*x)/(l[2]+np.exp(-l[3]*x))**2
	
	return y 

def SecDerSigmoid(l,x):

	first = -l[3]*np.exp(-l[3]*x)/(l[2]+np.exp(-l[3]*x))**2

	second = 2*l[3]*np.exp(-2*l[3]*x)/(l[2]+np.exp(-l[3]*x))**3

	y = l[1]*l[3]*(first + second)

	return y

def ParDerSigmoid(l,x):

	a = np.ones(x.shape)

	b = 1./(l[2]+np.exp(-l[3]*x))

	c = - l[1]/(l[2]+np.exp(-l[3]*x))**2

	d = l[1]*l[3]*np.exp(-l[3]*x)/(l[2]+np.exp(-l[3]*x))**2

	return [a,b,c,d]

def CosineBasis(K,T,dt,a=1.8,c=0.): #cosine function on logarithm scale. truncated.
	
	F = np.zeros((K,T/dt),dtype='float') 
	I = np.arange(T/dt)*dt

	for k in range(K): 

		lb = math.exp((k/2.-1)*math.pi/a) - c
		ub = math.exp((k/2.+1)*math.pi/a) - c

		cos_on_log = np.cos(a*np.log(I[(I>=lb)&(I<ub)]+c)-k*0.5*math.pi)

		F[k,(I>=lb)&(I<ub)] = 0.5*(1 + cos_on_log)

	return F

def NaturalSpline(knots,bnds):
	
	dv = 0.00001*(bnds[1]-bnds[0])

	F = np.zeros((len(knots),100000.),dtype='float')

	v = np.arange(bnds[0],bnds[1],dv)

	F[0,:] = 1.

	F[1,:] = v

	for i in range(2,len(knots)):

		tmp1 = (v-knots[i-2])**3
		tmp1[tmp1<0.] = 0.

		tmp2 = (v-knots[-1])**3
		tmp2[tmp2<0.] = 0.

		tmp3 = (v-knots[-2])**3
		tmp3[tmp3<0.] = 0.		

		dk = (tmp1 - tmp2)*(1./(knots[-1]-knots[i-2]))

		dkm1 = (tmp3 - tmp2)*(1./(knots[-1]-knots[-2]))

		F[i,:] = dk - dkm1

	return F

def Tents(knots,bnds,total_length):#uses knots and bnds to keep same syntax as splines

	Nb = len(knots)+2

	delta = total_length/(Nb-2)

	F = np.zeros((len(knots)+1,total_length),dtype='float')

	F[0,:delta] = np.arange(delta,0,-1)

	F[-1,-delta:] = np.arange(delta)

	for i in range(1,Nb-2,1):

		F[i,delta*i:delta*(i+1)] = np.arange(delta,0,-1)
		F[i,delta*(i-1):delta*i] = np.arange(delta)

	return F

def DerTents(knots,bnds,total_length):#derivative of tent basis functions.for gradient

	F = np.zeros((len(knots)+1,total_length),dtype='float')

	dv = (bnds[1] - bnds[0])*0.00001

	Nb = len(knots)+2

	delta = total_length/(Nb-2)

	F[0,:delta] = -1./dv
	F[-1,-delta:] = 1./dv

	for i in range(1,Nb-2,1):

		F[i,delta*i:delta*(i+1)] = -1./dv
		F[i,delta*(i-1):delta*i] = 1./dv

	return F

def SecDerTents(knots,bnds,total_length):

	F = np.zeros((len(knots)+1,total_length),dtype='float')

	return F	

def DerNaturalSpline(knots,bnds):

	dv = 0.001*(bnds[1]-bnds[0])

	F = np.zeros((len(knots),1000.),dtype='float')

	v = np.arange(bnds[0],bnds[1],dv)

	F[0,:] = 0.

	F[1,:] = 1.

	for i in range(2,len(knots)):

		tmp1 = 3*(v-knots[i-2])**2
		tmp1[tmp1<0.] = 0.

		tmp2 = 3*(v-knots[-1])**2
		tmp2[tmp2<0.] = 0.

		tmp3 = 3*(v-knots[-2])**2
		tmp3[tmp3<0.] = 0.		

		dk = (tmp1 - tmp2)*(1./(knots[-1]-knots[i-2]))

		dkm1 = (tmp3 - tmp2)*(1./(knots[-1]-knots[-2]))

		F[i,:] = dk - dkm1

	return F

def SplineParamsforId(knots,bnds):

	F = NaturalSpline(knots,bnds)

	dv = 0.00001*(bnds[1]-bnds[0])

	v = np.arange(bnds[0],bnds[1],dv)

	params = np.dot(np.linalg.inv(np.dot(F,F.transpose())),np.dot(F,v))
	
	return params

def Ker2Param(ker,basis): #CRUCIAL. 

	B = basis

	Nb = np.shape(B)[0]

	Nparam = np.shape(ker)[0]*np.shape(B)[0]
	
	paramKer = np.zeros((Nparam),dtype='float')

	for i in range(np.shape(ker)[0]):

		bbtm1 = np.linalg.inv(np.dot(B,B.transpose()))
	
		yxt = np.dot(ker[i,:],B.transpose()) 

		paramKer[i*Nb:(i+1)*Nb] = np.dot(yxt,bbtm1) #analytic formula of RSS minimum.
	
	return paramKer




