import numpy as np
from scipy import signal
import copy
import matplotlib.pylab as plt
import math
import os

os.chdir('/home/bmarchan/LNLN/my_cython/')

from convolve import convolve_cython_wrapper as cv

os.chdir('/home/bmarchan/LNLN/')

def likelihood(state):

	Ns = float(len(state.output[0]))

	MP = state.membrane_potential # defined at the end of this file.

	indices = np.floor(np.array(state.output[0])/state.dt) #convert to time-step index. 
	indices = indices.astype('int')

	Nsteps = int(state.total_time/state.dt)
	
	LL = np.sum(MP[indices]) - state.dt*np.sum(np.exp(MP)) 
	#the +1 in convolve guarantees a high value of MP at spike time. 

	LL = (1./(math.log(2.)*Ns))*(LL - Ns*(math.log(Ns/Nsteps) -1. ) )

	return LL #log-likelihood

def gradient_NL(state): #first of the gradient.

	Ns = float(len(state.output[0]))
	Nsteps = int(state.total_time/state.dt)
	Nb = state.basisNL[0].shape[0] #number of basis functions.
	dt = state.dt

	dim = state.paramNL.size

	gr_NL = np.zeros((dim,Nsteps),dtype='float')

	MP = state.membrane_potential 
	MP12 = state.sub_membrane_potential #contains membrane potential in group befo

	for g in range(state.Ng): #loop over compartments/groups

		u = MP12[g,:] 
			
		Nbnl = state.basisNL[g].shape[0]
		NL = np.dot(state.paramNL[g*Nbnl:(g+1)*Nbnl],state.basisNL[g]) # (1,100000)

		for i in range(Nb-1): #for loop to create a stack of NB times MP12[g,:]

			u = np.vstack((MP12[g,:],u))

		gr_NL[g*Nb:(g+1)*Nb,:] = applyNL_2d(state.basisNL[g],u,state.bnds[g]) #apply NL to 

	sptimes = np.floor(np.array(state.output[0])/state.dt) #conversion to timestep
	sptimes = sptimes.astype('int') #has to be int to be an array of indices.
	lambd = np.exp(MP) #MP contains the threshold. 

	gr_NL = np.sum(gr_NL[:,sptimes],axis=1) - dt*np.sum(gr_NL*lambd,axis=1) 

	#Before summation, gr_NL is the gradient of the membrane potential.

	return (1./(math.log(2.)*Ns))*gr_NL

def hessian_NL(state): 

	Ns = float(len(state.output[0]))
	Nsteps = int(state.total_time/state.dt) #Total number of time-steps.
	Nb = state.basisNL[0].shape[0] #number of basis functions.

	he_NL = np.zeros((Nb*state.Ng,Nb*state.Ng),dtype='float') 

	MP = state.membrane_potential #Membrane potential. Contains threshold, so log o
	MP12 = state.sub_membrane_potential # Memb. pot. before NL in compartments.
	
	for g in range(state.Ng): 
		for h in range(state.Ng): #Double for-loops over compartments.

			if g>=h: #so that computations are not carried out twice.

				ug = MP12[g,:] #need it to "stack".
				uh = MP12[h,:]

				for i in range(Nb-1): #for loop to create a stack of NB times MP12[g,:]
					ug = np.vstack((MP12[g,:],ug))
					uh = np.vstack((MP12[h,:],uh))

				ug = applyNL_2d(state.basisNL[g],ug,state.bnds[g]) 
				uh = applyNL_2d(state.basisNL[g],uh,state.bnds[g])

				uht = uh.transpose()

				lamb = np.exp(MP)

				m = np.dot(ug*lamb,uht) #gives a (Nb,Nb) matrix.

				he_NL[g*Nb:(g+1)*Nb,h*Nb:(h+1)*Nb] = - state.dt*m

	he_NL = 0.5*(he_NL+he_NL.transpose()) #because hessian is symetric.

	return (1./(math.log(2.)*Ns))*he_NL

def applyNL(NL,u,bnds): #crucial piece of code to apply NL to membrane potential.

	dv = (bnds[1] - bnds[0])*0.00001 

	u = (u-bnds[0])/dv
	u = np.floor(u)
	u = u.astype('int') #indices need to be recentered. 0mV -> 50000 -> 0mV
	
	u[u>99999] = 99999.
	u[u<0] = 0
	
	u = NL[u] # The values in the NL array are in mV.

	return u

def applyNL_2d(NL,u,bnds): #same thing, but when dimensions are different.

	dv = (bnds[1] - bnds[0])*0.00001

	u = (u-bnds[0])/dv
	u = np.floor(u)
	u = u.astype('int') #need to recenter.

	u[u<0] = 0.
	u[u>99999] = 99999

	if len(u.shape)==1: #if u is 1D but NL 2D (basis functions for instance)

		res = np.zeros((np.shape(NL)[0],np.size(u)),dtype='float') #res = result.

		for i in range(np.shape(NL)[0]):

			res[i,:] = NL[i,u] #the values in this array are in mV already.

	else: #if u is 2D and NL too.
		
		res = np.zeros((np.shape(NL)[0],np.shape(u)[-1]),dtype='float')

		for i in range(np.shape(u)[0]):

			res[i,:] = NL[i,u[i,:]] #mV

	return res

def gradient_ker(state): 

	Ns = float(len(state.output[0]))
	Nb = state.basisKer.shape[0]  #number of basis functions for kernels/PSP.
	Nsteps = int(state.total_time/state.dt) #total number of time steps.
	Nneur  = int(state.N/state.Ng) # Number of presyn. neurons in a compartments.
	N_ASP = state.basisASP.shape[0] # number of basis functions for ASP.
	Nbnl = state.basisNL[0].shape[0] #number of basis functions for NL
	dt = state.dt
	output = state.output

	gr_ker = np.zeros((state.Ng*Nb*Nneur+N_ASP+1,Nsteps),dtype='float')
	#PSP kernels + ASP + threshold.

	MP12 = state.sub_membrane_potential #before NL
	MP = state.membrane_potential #after NL and solved
	lamb = np.exp(MP) #firing rate (MP contains threshold)
	
	for g in range(state.Ng): #loop over compartments.

		Basis = state.basisKer 

		X = cv(state.input[g*Nneur:(g+1)*Nneur],Basis,state)

		nlDer = np.dot(state.paramNL[g*Nbnl:(g+1)*Nbnl],state.basisNLder[g]) 

		#need derivative of non-linearity.

		gr_ker[g*Nb*Nneur:(g+1)*Nb*Nneur,:] = X*applyNL(nlDer,MP12[g,:],state.bnds[g])

	gr_ker[state.Ng*Nb*Nneur:-1,:] = - cv(output,state.basisASP,state)

	sptimes = np.floor(np.array(state.output[0])/dt) #no +1 or -1 here. it is in 									#cv(-), and MembPot(-)
	sptimes = sptimes.astype('int')

	gr_ker = np.sum(gr_ker[:,sptimes],axis=1) - dt*np.sum(gr_ker*lamb,axis=1)
	
	gr_ker[-1] = - len(state.output[0]) + dt*np.sum(lamb)

	return (1./(math.log(2.)*Ns))*gr_ker

def hessian_ker(state):

	Ns = float(len(state.output[0]))
	Nb = state.basisKer.shape[0]
	Nsteps = int(state.total_time/state.dt)
	Nneur = int(state.N/state.Ng)
	N_ASP = state.basisASP.shape[0]
	Nbnl = state.basisNL[0].shape[0]
	Ng = state.Ng

	Hess_ker = np.zeros((Ng*Nneur*Nb+N_ASP+1,Ng*Nneur*Nb+N_ASP+1),dtype='float')

	MP12 = state.sub_membrane_potential
	MP = state.membrane_potential
	output = state.output		
	Basis = state.basisKer 
	lamb = np.exp(MP)
	
	for g in range(state.Ng):

		param = state.paramNL[g*Nbnl:(g+1)*Nbnl]
		basisder = state.basisNLder[g]
		basissec = state.basisNLSecDer[g]
		nlDer = np.dot(param,basisder)
		nlSecDer = np.dot(param,basissec)

		X1 = cv(state.input[g*Nneur:(g+1)*Nneur],Basis,state)
		
		v = applyNL(nlDer,MP12[g,:],state.bnds[g])
		u = applyNL(nlSecDer,MP12[g,:],state.bnds[g])

		X3 = cv(output,state.basisASP,state)

		Halgam = state.dt*np.dot(X1*lamb*v,X3.transpose())	

		Halthet = state.dt*np.sum(X1*v*lamb,axis=1)

		sptimes = np.floor(np.array(output[0])/state.dt)
		sptimes = sptimes.astype('int')

		Hspik = state.dt*np.dot(X1[:,sptimes]*u[sptimes],X1[:,sptimes].transpose())

		Hnlder = np.dot(X1*lamb*(v**2),X1.transpose())

		Hnlsecder = np.dot(X1*lamb*u,X1.transpose())

		Hnosp = state.dt*(Hnlder + Hnlsecder)

		Halpha = Hspik - Hnosp

		Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,-1] = Halthet

		Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,g*Nneur*Nb:(g+1)*Nneur*Nb] = Halpha

		Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP)] = Halgam

		for h in range(state.Ng):

			if g<>h:

		  		param1 = state.paramNL[g*Nbnl:(g+1)*Nbnl]
				param2 = state.paramNL[h*Nbnl:(h+1)*Nbnl]
	
				nlDer1 = np.dot(param1,basisder)
				nlDer2 = np.dot(param2,basisder)

				u = applyNL(nlDer1,MP12[g,:],state.bnds[g])
				v = applyNL(nlDer2,MP12[h,:],state.bnds[h])

				X1 = cv(state.input[g*Nneur:(g+1)*Nneur],Basis,state)
				X2 = cv(state.input[h*Nneur:(h+1)*Nneur],Basis,state)

				Halbet = - state.dt*np.dot(X1*u*v*lamb,X2.transpose())

				Hess_ker[g*Nneur*Nb:(g+1)*Nneur*Nb,h*Nneur*Nb:(h+1)*Nneur*Nb] = Halbet

	Hess_ker[-1,-1] = -state.dt*np.sum(np.exp(MP))

	Hgamthet = -state.dt*np.sum(X3*lamb,axis=1)

	Hgam = -state.dt*np.dot(X3*lamb,X3.transpose())

	Hess_ker[Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP),-1] = Hgamthet

	Hess_ker[Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP),Ng*Nneur*Nb:(Ng*Nneur*Nb+N_ASP)] = Hgam

	Hess_ker = 0.5*(Hess_ker.transpose() + Hess_ker)	

	return (1./(math.log(2.)*Ns))*Hess_ker

def subMembPot(state,string): #membrane potential before NL.

	if string=='training':

		Nsteps = int(state.total_time/state.dt) #total number of time steps.

	elif string=='test':

		Nsteps = int(state.total_time_test/state.dt)

	MP12 = np.zeros((state.Ng,Nsteps),dtype='float')

	Nneur = int(state.N/state.Ng) #number of neurons in compartment.
	Nb = state.basisKer.shape[0] #number of basis function for kernels.

	for g in range(state.Ng):

		Basis = state.basisKer

		if string=='training':

			inp = state.input[g*Nneur:(g+1)*Nneur]

		elif string=='test':

			inp = state.input_test[g*Nneur:(g+1)*Nneur]

		X = cv(inp,Basis,state)[:,:Nsteps] #+1 in there.

		MP12[g,:] = np.dot(state.paramKer[g*Nneur*Nb:(g+1)*Nneur*Nb],X)

	return MP12

def MembPot(state):

	Nsteps = int(state.total_time/state.dt) # total number of time-steps.

	MP = np.zeros(Nsteps)
	Nneurons = int(state.N/state.Ng) # number of neurons in compartment.
	Nbnl = np.shape(state.basisNL[0])[0] # number of basis functions for NL.

	ParKer = state.paramKer

	MP12 = state.sub_membrane_potential

	for g in range(state.Ng): #loop over compartments. 

		Mp_g = MP12[g,:] 

		F = state.basisNL[g] 
		NL = np.dot(state.paramNL[g*Nbnl:(g+1)*Nbnl],F)

		dv = (state.bnds[g][1] - state.bnds[g][0])*0.00001

		Mp_g = (Mp_g-state.bnds[g][0])/dv
		Mp_g = np.floor(Mp_g)
		Mp_g = Mp_g.astype('int')

		Mp_g[Mp_g>99999] = 99999
		Mp_g[Mp_g<0] = 0

		Mp_g = NL[Mp_g] #NL is an array with mV.

		MP = MP + Mp_g

	X = cv(state.output,state.basisASP,state) # +1 in there.

	Nb = np.shape(X)[0] 

	MP = MP - np.dot(ParKer[(-Nb-1):-1],X) - ParKer[-1]

	return MP


