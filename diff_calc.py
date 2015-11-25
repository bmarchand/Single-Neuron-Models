import numpy as np
from scipy import signal
import copy
import matplotlib.pylab as plt

def convolve(spike_train,basis,state):
	"""Each spike train has to be convoluted with all the basis functions. 
	spike_train is a list of lists of spike times. basis is a numpy array 
	of dimensions (number_of_basis_functions,length_ker_in_ms/dt). 
	state is included in case we need dt or something else."""

	Nsteps = int(state.total_time/state.dt)

	Nbasis = copy.copy(np.shape(basis)[0])

	ST = np.zeros((len(spike_train),Nsteps),dtype='float')

	lent = len(spike_train)

	for i in range(lent):

		indices = np.around(np.array(spike_train[i])*(1./state.dt))+[1.]
		indices = indices.astype('int')
		indices = indices[indices<Nsteps]

		ST[i,indices] = 1.

	X = np.zeros((lent*Nbasis,Nsteps),dtype='float')

	for i in range(np.shape(basis)[0]):

		exp_i = np.atleast_2d(basis[i,:])
		X[lent*i:lent*(i+1),:] = signal.fftconvolve(ST,exp_i)[:,:Nsteps]
	
	return X
	

def likelihood(state):

	MP = MembPot(state)

	indices = np.around(np.array(state.output))
	indices = indices.astype('int')

	LL = np.sum(MP[indices]) - state.dt*np.sum(np.exp(MP))

	return LL

def gradient_NL(state):

	Nsteps = int(state.total_time/state.dt)
	Nb = np.shape(state.basisNL)[0]
	dt = state.dt

	gr_NL = np.zeros((state.Ng*Nb,Nsteps),dtype='float')

	MP = MembPot(state)
	MP12 = subMembPot(state)

	for g in range(state.Ng):

		u = MP12[g,:]

		for i in range(Nb-1):
			u = np.vstack((MP12[g,:],u))

		gr_NL[g*Nb:(g+1)*Nb,:] = applyNL_2d(state.basisNL,u,state)

	sptimes = np.around(np.array(state.output[0])/state.dt)
	sptimes = sptimes.astype('int')
	lambd = np.exp(MP)

	gr_NL = np.sum(gr_NL[:,sptimes],axis=1) - dt*np.sum(gr_NL*lambd,axis=1)

	return gr_NL

def hessian_NL(state):

	Nsteps = int(state.total_time/state.dt)
	Nb = np.shape(state.basisNL)[0]

	he_NL = np.zeros((Nb*state.Ng,Nb*state.Ng,Nsteps),dtype='float')

	MP = MembPot(state)
	MP12 = subMembPot(state)
	
	for g in range(state.Ng):
		for h in range(state.Ng):

			if g>=h:

				ug = np.atleast_2d(MP12[g,:])
				uh = np.atleast_2d(MP12[h,:])
		
				ug = np.repeat(ug,Nb,axis=0)
				uh = np.repeat(uh,Nb,axis=0)

				ug = applyNL_2d(state.basisNL,ug,state)
				uh = applyNL_2d(state.basisNL,uh,state)
				uht = uh.transpose()

				m = np.dot(ug,uht)
				ddot = np.atleast_3d(m)
				ddot = np.repeat(ddot,Nsteps,axis=2)

				he_NL[g*Nb:(g+1)*Nb,h*Nb:(h+1)*Nb,:] = ddot*np.exp(MP)

	he_NL = -state.dt*np.sum(he_NL,axis=2)

	he_NL = 0.5*(he_NL+he_NL.transpose())

	return he_NL

def applyNL(NL,u,state):

	dv = (state.bnds[1] - state.bnds[0])*0.00001

	h = copy.copy(np.histogram(u))

	u = u/dv
	u = np.around(u)
	u = u.astype('int') + 50000
	u = NL[u]/dv

	h1 = np.histogram(u)

	return u

def applyNL_2d(NL,u,state):

	dv = (state.bnds[1] - state.bnds[0])*0.00001

	u = u/dv
	u = np.around(u)
	u = u.astype('int') + 50000

	if len(u.shape)==1:

		res = np.zeros((np.shape(NL)[0],np.size(u)),dtype='float')

		for i in range(np.shape(NL)[0]):

			res[i,:] = NL[i,u]

	else:
		
		res = np.zeros((np.shape(NL)[0],np.shape(u)[-1]),dtype='float')

		for i in range(np.shape(u)[0]):

			res[i,:] = NL[i,u[i,:]]

	return res

def gradient_ker(state):

	Nb = np.shape(state.basisKer)[0]
	Nsteps = int(state.total_time/state.dt)
	Nneur  = int(state.N/state.Ng)
	N_ASP = len(state.knots_ASP)+1
	Nbnl = np.shape(state.basisNL)[0]
	dt = state.dt
	output = state.output

	gr_ker = np.zeros((state.Ng*Nb*Nneur+N_ASP+1,Nsteps),dtype='float')

	MP12 = subMembPot(state)
	MP = MembPot(state)
	lamb = np.exp(MP)
	
	for g in range(state.Ng):

		Basis = state.basisKer

		X = convolve(state.input[g*Nneur:(g+1)*Nneur],Basis,state)

		nlDer = np.dot(state.paramNL[g*Nbnl:(g+1)*Nbnl],state.basisNLder)

		gr_ker[g*Nb*Nneur:(g+1)*Nb*Nneur,:] = X*applyNL(nlDer,MP12[g,:],state)

	gr_ker[state.Ng*Nb*Nneur:-1,:] = - convolve(output,state.basisASP,state)

	sptimes = np.around(np.array(state.output[0])/dt)+[1]
	sptimes = sptimes[sptimes<Nsteps]
	sptimes = sptimes.astype('int')

	OST = np.zeros(Nsteps)

	OST[sptimes] = 1.

	gr_ker = np.sum(gr_ker[:,sptimes],axis=1) - dt*np.sum(gr_ker*lamb,axis=1)

	gr_ker[-1] = - len(state.output) + dt*np.sum(lamb)

	return gr_ker

def hessian_ker(state):

	Nb = np.shape(state.basisKer)[0]
	Nsteps = int(state.total_time/state.dt)
	Nneur = int(state.N/state.Ng)
	N_ASP = len(state.knots_ASP)+1
	Nbnl = np.shape(state.basisNL)[0]

	gr_ker = np.zeros((state.Ng*Nb*Nneur+N_ASP+1,Nsteps),dtype='float')

	MP12 = subMembPot(state)
	MP = MembPot(state)
	output = state.output
	
	for g in range(state.Ng):

		Basis = state.basisKer

		X = convolve(state.input[g*Nneur:(g+1)*Nneur],Basis,state)
		
		nlDer = np.dot(state.paramNL[g*Nbnl:(g+1)*Nbnl],state.basisNLder)
		
		gr_ker[g*Nb*Nneur:(g+1)*Nb*Nneur,:] = X*applyNL(nlDer,MP12[g,:],state)

	gr_ker[state.Ng*Nb*Nneur:-1] = - convolve(output,state.basisASP,state)

	gr_ker[-1] = - len(state.output) + state.dt*np.sum(np.exp(MP))

	Hess = -state.dt*np.dot(gr_ker*np.exp(MP),gr_ker.transpose())

	return Hess

def subMembPot(state):

	Nsteps = int(state.total_time/state.dt)

	MP12 = np.zeros((state.Ng,Nsteps),dtype='float')

	Nneur = int(state.N/state.Ng)
	Nb = state.N_cos_bumps

	for g in range(state.Ng):

		Basis = state.basisKer

		X = convolve(state.input[g*Nneur:(g+1)*Nneur],Basis,state)

		MP12[g,:] = np.dot(state.paramKer[g*Nneur*Nb:(g+1)*Nneur*Nb],X)

	return MP12

def MembPot(state):

	Nsteps = int(state.total_time/state.dt)

	ParKer = state.paramKer

	MP = np.zeros(Nsteps)

	Nneurons = int(state.N/state.Ng)
	Nbnl = np.shape(state.basisNL)[0]

	MP12 = subMembPot(state)

	for g in range(state.Ng):

		Mp_g = MP12[g,:]

		F = state.basisNL

		NL = np.dot(F.transpose(),state.paramNL[g*Nbnl:(g+1)*Nbnl])

		dv = (state.bnds[1] - state.bnds[0])*0.00001

		h0 = copy.copy(np.histogram(Mp_g))

		Mp_g = Mp_g/dv

		Mp_g = np.around(Mp_g)
		
		Mp_g = Mp_g.astype('int') + 50000

		Mp_g = NL[Mp_g]

		h1 = np.histogram(Mp_g)

		MP = MP + Mp_g

	X = convolve(state.output,state.basisASP,state)

	Nb = np.shape(X)[0]

	MP = MP - np.dot(ParKer[(-Nb-1):-1],X) - ParKer[-1]

	return MP


