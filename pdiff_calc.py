import numpy as np
from scipy import signal
import copy
import matplotlib.pylab as plt
from scipy import weave
from scipy.weave import converters
import os
import convolves as conv
import functions as fun
import math

os.chdir('/home/bmarchan/LNLN/my_cython/')

from convolve import convolve_cython_wrapper as cv

os.chdir('/home/bmarchan/LNLN/')

def likelihood(state):

	MP = state.membrane_potential # defined at the end of this file.

	indices = np.around(np.array(state.output[0])/state.dt) #convert to time-step index. 
	indices = indices.astype('int')

	Nsteps = int(state.total_time/state.dt)
	
	LL = np.sum(MP[indices]) - state.dt*np.sum(np.exp(MP)) 
	#the +1 in convolve guarantees a high value of MP at spike time. 

	Ns = float(len(state.output[0]))

	Nsteps = state.total_time/state.dt

	LL = 1/(math.log(2.)*Ns)*(LL - Ns*(math.log(Ns/Nsteps) -1.) )

	return LL #log-likelihood

def gradient_NL(state): #first of the gradient.

	Ns = float(len(state.output[0]))

	Nsteps = int(state.total_time/state.dt)
	dt = state.dt

	npnl = state.paramNL.shape[1]

	Np = np.size(state.paramNL)

	gr_NL = np.zeros((Np,Nsteps),dtype='float')

	MP = state.membrane_potential 
	MP12 = state.sub_membrane_potential #contains membrane potential in group before NL

	for g in range(state.Ng): #loop over compartments/groups

		u = MP12[g,:] 

		dabcd = fun.ParDerSigmoid(state.paramNL[g],u)

		gr_NL[g*npnl:(g+1)*npnl,:] = np.array(dabcd)

	sptimes = np.around(np.array(state.output[0])/state.dt) #conversion to timestep
	sptimes = sptimes.astype('int') #has to be int to be an array of indices.
	lambd = np.exp(MP) #MP contains the threshold. 

	gr_NL = np.sum(gr_NL[:,sptimes],axis=1) - dt*np.sum(gr_NL*lambd,axis=1) 

	#Before summation, gr_NL is the gradient of the membrane potential.

	return (1./(math.log(2)*Ns))*gr_NL

def hessian_NL(state): 

	Ns = float(len(state.output[0]))

	Nsteps = int(state.total_time/state.dt) #Total number of time-steps.

	npnl = state.paramNL.shape[1]

	Np = state.paramNL.size

	he_NL = np.zeros((Np,Np),dtype='float') 

	MP = state.membrane_potential #Membrane potential. Contains threshold, so log of lambda.
	MP12 = state.sub_membrane_potential # Memb. pot. before NL in compartments.
	
	for g in range(state.Ng): 
		for h in range(state.Ng): #Double for-loops over compartments.

			if g>=h: #so that computations are not carried out twice.

				ug = MP12[g,:] #need it to "stack".
				uh = MP12[h,:]

				dabcdg = np.array(fun.ParDerSigmoid(state.paramNL[g],ug))
				dabcdh = np.array(fun.ParDerSigmoid(state.paramNL[h],uh))

				lamb = np.exp(MP)

				m = np.dot(dabcdg*lamb,dabcdh.transpose()) #gives a (Nb,Nb) matrix.

				he_NL[g*npnl:(g+1)*npnl,h*npnl:(h+1)*npnl] = - state.dt*m

	he_NL = 0.5*(he_NL+he_NL.transpose()) #because hessian is symetric.

	return (1./(math.log(2)*Ns))*he_NL

def gradient_ker(state): 

	Ns = float(len(state.output[0]))
	Nb = np.shape(state.basisKer)[0]  #number of basis functions for kernels/PSP.
	Nsteps = int(state.total_time/state.dt) #total number of time steps.
	Nneur  = int(state.N/state.Ng) # Number of presyn. neurons in a compartments.
	N_ASP = state.basisASP.shape[0] # number of basis functions for ASP.
	dt = state.dt
	output = state.output

	gr_ker = np.zeros((state.Ng*Nb*Nneur+N_ASP+1,Nsteps),dtype='float')
	#PSP kernels + ASP + threshold.

	MP12 = state.sub_membrane_potential #before NL
	MP = state.membrane_potential #after NL and solved
	lamb = np.exp(MP) #firing rate (MP contains threshold)
	
	for g in range(state.Ng): #loop over compartments.

		Basis = state.basisKer 

		from convolve import convolve_cython_wrapper as cv

		X = cv(state.input[g*Nneur:(g+1)*Nneur],Basis,state)

		nlDer_to_u = fun.DerSigmoid(state.paramNL[g],MP12[g,:])

		#need derivative of non-linearity.

		gr_ker[g*Nb*Nneur:(g+1)*Nb*Nneur,:] = X*nlDer_to_u

	gr_ker[state.Ng*Nb*Nneur:-1,:] = - cv(output,state.basisASP,state)

	sptimes = np.around(np.array(state.output[0])/dt) #no +1 or -1 here. it is in   													#cv(-), and MembPot(-)
	sptimes = sptimes.astype('int')

	gr_ker = np.sum(gr_ker[:,sptimes],axis=1) - dt*np.sum(gr_ker*lamb,axis=1)
	
	gr_ker[-1] = - len(state.output[0]) + dt*np.sum(lamb)

	return (1./(math.log(2)*Ns))*gr_ker

def hessian_ker(state):

	Ns = float(len(state.output[0]))
	Nb = state.basisKer.shape[0]
	Nsteps = int(state.total_time/state.dt)
	Nneur = int(state.N/state.Ng)
	N_ASP = state.basisASP.shape[0]
	Ng = state.Ng

	Hess_ker = np.zeros((Ng*Nneur*Nb+N_ASP+1,Ng*Nneur*Nb+N_ASP+1),dtype='float')

	MP12 = state.sub_membrane_potential
	MP = state.membrane_potential
	output = state.output		
	Basis = state.basisKer 
	lamb = np.atleast_2d(np.exp(MP)).transpose()
	
	for g in range(state.Ng):

		X1 = cv(state.input[g*Nneur:(g+1)*Nneur],Basis,state)

		v = fun.DerSigmoid(state.paramNL[g],MP12[g,:])

		v = np.atleast_2d(v).transpose()

		u = fun.SecDerSigmoid(state.paramNL[g],MP12[g,:])
		u = np.atleast_2d(u).transpose()

		X3 = cv(output,state.basisASP,state)#good

		Halgam = state.dt*np.dot(X1*np.exp(MP),X3.transpose()*v)	

		Halthet = state.dt*np.sum(X1.transpose()*v*lamb,axis=0)

		sptimes = np.around(np.array(output[0])/state.dt)
		sptimes = sptimes.astype('int')

		X1u = X1.transpose()*u#good

		Hspik = state.dt*np.dot(X1[:,sptimes],X1u[sptimes,:])#good

		Hnlder = np.dot(X1,X1.transpose()*(v**2)*lamb)#good

		Hnlsecder = np.dot(X1,X1.transpose()*u*lamb)#good

		Hnosp = state.dt*(Hnlder + Hnlsecder)#good

		Halpha = Hspik - Hnosp#good

		Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,-1] = Halthet

		Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,g*Nneur*Nb:(g+1)*Nneur*Nb] = Halpha#good

		Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP)] = Halgam

		for h in range(state.Ng):

			if g<>h:

				u = fun.DerSigmoid(state.paramNL[g],MP12[g,:])#good
				v = fun.DerSigmoid(state.paramNL[h],MP12[h,:])#good

				u = np.atleast_2d(u).transpose()#good
				v = np.atleast_2d(v).transpose()#good

				X1 = cv(state.input[g*Nneur:(g+1)*Nneur],Basis,state)#good
				X2 = cv(state.input[h*Nneur:(h+1)*Nneur],Basis,state)#good

				Halbet = - state.dt*np.dot(X1,X2.transpose()*u*v*lamb)#good

				Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,h*Nneur*Nb:(h+1)*Nneur*Nb] = Halbet

	Hess_ker[-1,-1] = -state.dt*np.sum(np.exp(MP))#good

	Hgamthet = -state.dt*np.sum(X3*np.exp(MP),axis=1)

	Hgam = -state.dt*np.dot(X3*np.exp(MP),X3.transpose())#good

	Hess_ker[Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP),-1] = Hgamthet

	Hess_ker[Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP),Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP)] = Hgam

	Hess_ker = 0.5*(Hess_ker.transpose() + Hess_ker)	

	return (1./(math.log(2)*Ns))*Hess_ker

def subMembPot(state): #membrane potential before NL.

	Nsteps = int(state.total_time/state.dt) #total number of time steps.

	MP12 = np.zeros((state.Ng,Nsteps),dtype='float')

	Nneur = int(state.N/state.Ng) #number of neurons in compartment.
	Nb = state.basisKer.shape[0] #number of basis function for kernels.

	for g in range(state.Ng):

		Basis = state.basisKer

		X = cv(state.input[g*Nneur:(g+1)*Nneur],Basis,state) #+1 in there.

		MP12[g,:] = np.dot(state.paramKer[g*Nneur*Nb:(g+1)*Nneur*Nb],X)

	return MP12

def PMembPot(state):

	Nsteps = int(state.total_time/state.dt)
	ParKer = state.paramKer

	MP = np.zeros(Nsteps)
	Nneurons = int(state.N/state.Ng)

	MP12 = state.sub_membrane_potential

	for g in range(state.Ng):

		Mp_g = MP12[g,:]

		l = state.paramNL[g]
	
		Mp_g = fun.sigmoid(l,Mp_g)

		MP = MP + Mp_g

	X = cv(state.output,state.basisASP,state) # +1 in there.

	Nb = np.shape(X)[0] 

	MP = MP - np.dot(ParKer[(-Nb-1):-1],X) - ParKer[-1]

	return MP


