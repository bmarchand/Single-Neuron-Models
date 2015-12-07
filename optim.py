import numpy as np
import functions as fun
from scipy import signal
import copy
import diff_calc as diff
import main
import matplotlib.pylab as plt
import math
import time

class State(main.TwoLayerModel,main.FitParameters,main.RunParameters):
#kind of the "microstate", as in statistical physics, of the model. The model itself
#only contains the model parameters and the fitting parameters. not the gradients
# nor the hessian.
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

		Np = int(np.size(Model.paramNL)*0.5) #number of basis functions per NL.

	def iter_ker(self,rate): 

		invH = np.linalg.inv(self.hessian_ker)
		
		self.paramKer = self.paramKer - rate*np.dot(invH,self.gradient_ker)

	def iter_NL(self):

		norm = math.sqrt(abs(np.sum(self.gradient_NL**2)))

		self.paramNL = self.paramNL + self.gradient_NL/norm

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

		fig = plt.figure()

		plt.ion()

		n0 = copy.copy(norm)

		norm_ker = abs(np.sum(state.gradient_ker**2))

		diff = 1.

		while diff > 0:

			l0 = copy.copy(state.likelihood)

			print "count ker:", cnt
			cnt = cnt + 1

			plt.plot(np.dot(state.paramKer[:5],state.basisKer))		

			plt.draw()			

			time.sleep(0.05)

			if cnt<15:
	
				rate = 0.2

			else:
				rate = 1.

			state.iter_ker(rate)

			state.update()

			diff = state.likelihood - l0
			
			norm_ker = abs(np.sum(state.gradient_ker**2))

			print state.likelihood, norm_ker

		diff = 1.

		cnt = 0.

		plt.close()
		plt.figure()
		plt.ion()

		while diff > 0.:

			plt.plot(np.dot(state.paramNL[:11],state.basisNL))
			plt.draw()

			cnt = cnt +1

			l0 = copy.copy(state.likelihood)

			state.iter_NL()
			state.update()

			norm_NL = abs(np.sum(state.gradient_NL**2))

			diff = state.likelihood - l0

			print "count NL: ",cnt

			print state.likelihood
	
	return state.paramNL,state.paramKer,state.likelihood

	
	
