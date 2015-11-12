import numpy as np
from scipy import signal
import copy

def convolve(spike_train,basis,model):
	"""Each spike train has to be convoluted with all the basis functions. spike_train is a list of lists of spike times. basis 
is a numpy array of dimensions (number_of_basis_functions,length_ker_in_ms/dt). model is included in case we need dt or something else."""

	Nsteps = int(model.total_time/model.dt)

	Nbasis = copy.copy(np.shape(basis)[0])

	ST = np.zeros((len(spike_train),Nsteps),dtype='float')

	for i in range(len(spike_train)):

		indices = np.around(spike_train[i]/model.dt)

		ST[i,indices] = 1.

	X = np.zeros((len(spike_train)*Nbasis,Nsteps),dtype='float')

	for i in range(np.shape(basis)[0]):

		X[len(spike_train)*i:len(spike_train)*(i+1),:] = signal.fftconvolve(ST,np.atleast_2d(basis[i,:]))
	
	return X
	

def likelihood(model):

	MP = MembPot(model)

	LL = np.sum(MP[np.around(model.output)]) - model.dt*np.sum(np.exp(MP))

	return LL

def gradient_NL(model):

	Nsteps = int(model.total_time/model.dt)
	Nb = np.shape(model.basisNL)[0]

	gradient_NL = np.zeros((model.Ng*Nb,Nsteps),dtype='float')

	MP = membranepot(model)
	MP12 = submembpot(model)

	for g in range(model.Ng):

		u = np.repeat(MP12[:,g],Nb,axis=0)

		gradient_NL[g*Nb:(g+1)*Nb,:] = applyNL(model.basisNL,u,model)

	sptimes = np.around(model.output/model.dt)

	gradient_NL = np.sum(gradient_NL[:,sptimes],axis=1) -model.dt*np.sum(gradient_NL*np.exp(MP),axis=1)

	return gradient_NL

def hessian_NL(model):

	Nsteps = int(model.total_time/model.dt)
	Nb = np.shape(model.basisNL)[0]

	hessian_NL = np.zeros((Nb*model.Ng,nb*model.Ng,Nsteps),dtype='float')

	MP = membranepot(model)
	MP12 = submembpot(model)
	
	for g in range(model.Ng):
		for h in range(model.Ng):

			if g>=h:
				ug = np.repeat(MP12[:,g],Nb,axis=0)
				uh = np.repeat(MP12[:,h],Nb,axis=0)
				ug = applyNL(model.basisNL,ug,model)
				uh = applyNL(model.basisNL,uh,model)

				hessian_NL[g*Nb:(g+1)*Nb,h*Nb:(h+1)*Nb,:] = np.dot(ug,uh.transpose())*np.exp(MP)

	hessian_NL = -model.dt*np.sum(hessian_NL,axis=2)

	hessian_NL = 0.5*(hessian_NL+hessian_NL.transpose())

	return hessian_NL

def applyNL(NL,u,model):

	dv = (model.bnds[1] - model.bnds[0])*0.001

	u = u/dv
	u = u.astype('int')
	u = NL[u]

	return u

def gradient_ker(model):

	Nb = np.shape(model.basisKer)[0]
	Nsteps = int(model.total_time/model.dt)
	Nneur = int(model.N/model.Ng)
	N_ASP = len(model.knots_ASP)
	nlDer = model.basisNLder

	gradient_ker = np.zeros((model.Ng*Nb*Nneur+N_ASP,Nsteps),dtype='float')

	MP12 = subMembPot(model)
	MP = membranepot(model)
	
	for g in range(model.Ng):

		Basis = model.basisKer

		X = convolve(model.input[g*Nneur:(g+1)*Nneur],Basis,model)

		gradient_ker[g*Nb*Nneur:(g+1)*Nb*Nneur,:] = X*applyNL(nlDer,MP12[:,g],model)

	gradient_ker[model.Ng*Nb*Nneur:] = - convolve(model.output,model.basisASP,model)

	sptimes = np.around(model.output/model.dt)

	gradient_ker = np.sum(gradient_ker[:,sptimes],axis=1) - model.dt*np.sum(gradient_ker*np.exp(MP),axis=1)

	return gradient_ker

def hessian_ker(model):

	Nb = np.shape(model.basisKer)[0]
	Nsteps = int(model.total_time/model.dt)
	Nneur = int(model.N/model.Ng)
	N_ASP = len(model.knots_ASP)
	nlDer = model.basisNLder

	gradient_ker = np.zeros((model.Ng*Nb*Nneur+N_ASP,Nsteps),dtype='float')

	MP12 = subMembPot(model)
	MP = membranepot(model)
	
	for g in range(model.Ng):

		Basis = model.basisKer

		X = convolve(model.input[g*Nneur:(g+1)*Nneur],Basis,model)

		gradient_ker[g*Nb*Nneur:(g+1)*Nb*Nneur,:] = X*applyNL(nlDer,MP12[:,g],model)

	gradient_ker[model.Ng*Nb*Nneur:] = - convolve(model.output,model.basisASP,model)

	Hess = -model.dt*np.dot(gradient_ker,np.exp(MP)*gradient_ker.transpose())

	return Hess

def subMembPot(model):

	Nsteps = int(model.total_time/model.dt)

	MP12 = np.zeros((Nsteps,model.Ng),dtype='float')

	Nneurons = int(model.N/model.Ng)

	for g in range(model.Ng):

		Basis = model.basisKer

		X = convolve(model.input[g*Nneurons:(g+1)*Nneurons],Basis,model)

		MP12[:,g] = np.dot(model.paramKer[g*model.N_cos_bump:(g+1)*model.N_cos_bump],X)

	return MP12

def MembPot(model):

	Nsteps = int(model.total_time/model.dt)

	MP = np.zeros(Nsteps)

	Nneurons = int(model.N/model.Ng)

	MP12 = submembpot(model)

	for g in range(model.Ng):

		Mp_g = MP12[:,g]

		F = model.basisNL

		NL = np.dot(F,model.paramNL)

		dv = (model.bnds[1] - model.bnds[0])*0.001

		Mp_g = Mp_g/dv

		MP_g = MP_g.astype('int')

		Mp_g = NL[Mp_g]
	
		MP = MP + MP_g

	X = convolve(model.output,model.basisASP,model)

	MP = MP - np.dot(X,model.paramASP) - model.paramKer[-1]

	return MP


