import numpy as np
import functions as fun
from scipy import signal
import copy
import diff_calc as diff
import main

class State(main.TwoLayerModel,main.FitParameters,main.RunParameters):

	def __init__(self,Model):

		self.input = Model.input
		self.output = Model.output

		self.gradient_ker = diff.gradient_ker(Model)
		self.gradient_nl = diff.gradient_NL(Model)

		self.hessian_ker = diff.hessian_ker(Model)
		self.hessian_nl = diff.hessian_NL(Model)

		self.paramKer = Model.paramKer
		self.paramNL = Model.paramNL

		self.likelihood = diff.likelihood(Model)

	def iter_ker(self):

		self.paramKer = self.paramKer - np.dot(np.linalg.inv(self.hessian_ker),self.gradient_ker)

	def iter_NL(self):

		self.paramNL = self.paramNL - np.dot(np.linalg.inv(hessian_nl),self.gradient_NL)

	def update(self):

		self.membrane_potential = diff.MembPot(self)

		self.likelihood = diff.likelihood(self)
		
		self.gradient_ker = diff.gradient_ker(self)
		self.hessian_ker = diff.hessian_ker(self)

		self.gradient_NL = diff.gradient_NL(self)
		self.hessian_NL = diff.hessian_NL(self)
		
def BlockCoordinateAscent(Model):

	err = 1000.

	state = State(Model)
	
	while err>Model.tol:

		L0 = copy.copy(state.likelihood)

		print state.paramKer

		state.iter_ker()

		print state.paramKer
		print state.gradient_ker

		state.update()

		err = state.likelihood - L0

		if err>Model.tol:

			L0 = copy.copy(state.likelihood)
		
			state.iter_NL()
			state.update()

			err = state.likelihood - L0
		
	return state.paramNL,state.paramKer,state.likelihood

	
	
