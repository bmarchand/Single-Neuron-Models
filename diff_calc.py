import numpy as np

def likelihood(model):
	
	Nsteps = int(model.total_time/model.dt)

	MP = np.zeros(Nsteps)

	for g in range(model.Ng):

		ST = np.zeros((int(N/Ng),Nsteps),dtype='float')

		for i in range(int(N/Ng)):

			indices = model.input[g*int(N/Ng)+i]

			ST[i,model.input]

		Mp_g = np.dot(X,model.paramKer[g*model.N_cos_bump:(g+1)*N_cos_bump])
