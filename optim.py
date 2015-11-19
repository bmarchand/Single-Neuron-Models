import numpy as np
import functions as fun
from scipy import signal
import copy
import diff_calc as diff
import main
import matplotlib.pylab as plt

class State(main.TwoLayerModel,main.FitParameters,main.RunParameters):

	def __init__(self,Model):

		self.input = Model.input
		self.output = Model.output

		self.gradient_ker = diff.gradient_ker(Model)
		self.gradient_NL = diff.gradient_NL(Model)

		self.hessian_ker = diff.hessian_ker(Model)
		self.hessian_NL = diff.hessian_NL(Model)

		self.paramKer = Model.paramKer
		self.paramNL = Model.paramNL

		self.likelihood = diff.likelihood(Model)

		Np = int(np.size(Model.paramNL)*0.5)

		self.NL = np.dot(self.paramNL[:Np],self.basisNL)

	def iter_ker(self):

		invH = np.linalg.inv(self.hessian_ker)

		self.paramKer = self.paramKer - np.dot(invH,self.gradient_ker)

	def iter_NL(self):

		invH = np.linalg.inv(hessian_nl)

		self.paramNL = self.paramNL - np.dot(invH,self.gradient_NL)

	def update(self):

		self.membrane_potential = diff.MembPot(self)

		Np = int(np.size(self.paramNL)*0.5)
		self.NL = np.dot(self.paramNL[:Np],self.basisNL)

		self.likelihood = diff.likelihood(self)
		
		self.gradient_ker = diff.gradient_ker(self)
		self.hessian_ker = diff.hessian_ker(self)

		self.gradient_NL = diff.gradient_NL(self)
		self.hessian_NL = diff.hessian_NL(self)
		
def BlockCoordinateAscent(Model):

	state = State(Model)

	norm = abs(np.sum(state.gradient_ker**2)+np.sum(state.gradient_NL**2))

	cnt = 0.
	
	while norm>Model.tol:

		n0 = copy.copy(norm)

		norm_ker = abs(np.sum(state.gradient_ker**2))

		while norm_ker > Model.tol:

			print "count ker:", cnt
			cnt = cnt + 1

			plt.plot(state.NL)
			plt.show()
	
			state.iter_ker()

			state.update()
			
			norm_ker = abs(np.sum(state.gradient_ker**2))

		norm_NL = abs(np.sum(state.gradient_NL**2))

		while norm_NL > Model.tol:

			state.iter_NL()
			state.update()

			norm_NL = abs(np.sum(state.gradient_NL**2))

		
	return state.paramNL,state.paramKer,state.likelihood

	
	
