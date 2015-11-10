import numpy as np
from scipy import signal

def likelihood(model):

	MP = MembPot(model) - model.paramKer[-1]

	LL = np.sum(MP[np.around(model.output)]) - model.dt*np.sum(np.exp(MP))

	return LL

def MembPot(model):

	Nsteps = int(model.total_time/model.dt)

	MP = np.zeros(Nsteps)

	for g in range(model.Ng):

		ST = np.zeros((int(N/Ng),Nsteps),dtype='float')

		X = np.zeros((model.N_cos_bump,Nsteps),dtype='float')

		for i in range(int(N/Ng)):

			indices = model.input[g*int(N/Ng)+i]

			indices = np.around(indices)

			ST[i,indices] = 1.

		ST = np.atleast_3d(ST)

		ST = np.repeat(ST,model.N_cos_bump,axis=2)

		Basis = model.basisKer

		Basis = np.reshape(Basis,(1,np.shape(Basis)[0],np.shape(Basis)[1]))

		X = signal.fftconvolve(ST,Basis)

		X = np.sum(X,axis=0)

		Mp_g = np.dot(model.paramKer[g*model.N_cos_bump:(g+1)*model.N_cos_bump],X)

		NL = np.dot(F,model.paramNL)

		dv = (model.bnds[1] - model.bnds[0])*0.001

		Mp_g = Mp_g/dv

		MP_g = MP_g.astype('int')

		Mp_g = NL[Mp_g]
	
		MP = MP + MP_g

		OST = np.zeros(Nsteps)

		indices = np.around(model.output)

		OST[indices] = 1.

		OST = np.atleast_2d(OST)

		OST = np.repeat(OST,len(model.knots_ASP),axis=1)

		Basis = model.basisASP

		X = signal.fftconvolve(OST,Basis)

		X = np.sum(X,axis=1)

		MP = MP - np.dot(X,model.paramASP)

	return MP

