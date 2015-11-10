import numpy as np
import functions as fun
from scipy import signal

class State(Model):

	def iter_ker(self):

		gradient_ker = np.zeros(np.shape(self.gradient),dtype='float')

		gradient_ker[:self.N_param_ker]

		self.paramKer = self.paramKer - np.dot(np.linalg.inv(self.hessian_ker),self.gradient_ker)

	def iter_NL(self):

		self.paramNL = self.paramNL - np.dot(np.linalg.inv(hessian),self.gradient_NL)

	def update(self,Model):

		self.membrane_potential = membrane_potential(self)

		self.likelihood = optim.likelihood(self)
		self.gradient = optim.gradient(self)
		self.hessian = optim.hessian(self)

def BlockCoordinateAscent(Model):

	err = 1000.

	state = State(Model)
	
	while err>Model.tol:

		L0 = copy.copy(state.likelihood)

		state.iter_ker()
		state.update()

		err = state.likelihood - L0

		if err>Model.tol:

			L0 = copy.copy(state.likelihood)
		
			state.iter_NL()
			state.update()

			err = state.likelihood - L0
		
	return state.paramNL,state.paramKer,state.likelihood

	
	