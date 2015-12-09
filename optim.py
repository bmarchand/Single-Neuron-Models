import numpy as np
import functions as fun
from scipy import signal
import copy
import pdiff_calc as diff
import main
import matplotlib.pylab as plt
from matplotlib import gridspec
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

		self.sub_membrane_potential = diff.subMembPot(Model)
		self.membrane_potential = diff.PMembPot(Model)
		

	def iter_ker(self,rate): 

		invH = np.linalg.pinv(self.hessian_ker)

		self.paramKer = self.paramKer - rate*np.dot(invH,self.gradient_ker)

	def iter_NL(self):

		invH = np.linalg.pinv(self.hessian_NL)

		for g in range(len(self.paramNL)):

			Np = len(self.paramNL[g])

			change = np.dot(invH,self.gradient_NL)[g*Np:(g+1)*Np]

			self.paramNL[g] = self.paramNL[g] - change

	def update(self):

		self.sub_membrane_potential = diff.subMembPot(self)
		self.membrane_potential = diff.PMembPot(self)

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

		gs = gridspec.GridSpec(4,1)

		fig1 = plt.figure()
		ax1 = fig1.add_subplot(gs[0,0])
		ax2 = fig1.add_subplot(gs[1,0])
		ax3 = fig1.add_subplot(gs[2,0])
		ax4 = fig1.add_subplot(gs[3,0])

		fig1.show()

		plt.draw()

		plt.ion()

		n0 = copy.copy(norm)

		norm_ker = math.sqrt(abs(np.sum(state.gradient_ker**2)))

		diff = 1.

		while norm_ker > 6.:

			l0 = copy.copy(state.likelihood)

			print "count ker:", cnt
			cnt = cnt + 1		

			ax1.plot(np.dot(state.paramKer[-10:-6],state.basisKer))
			ax2.plot(np.dot(state.paramKer[:4],state.basisKer))	
			ax3.plot(np.dot(state.paramKer[-6:-1],state.basisASP))		
			ax4.plot(1-np.exp(-np.exp(state.membrane_potential[:3000])))
			fig1.canvas.draw()	
			time.sleep(0.005)

			if cnt<10:
				rate = 0.8
			else:
				rate = 1.

			state.iter_ker(rate)
			state.update()

			diff = state.likelihood - l0
			
			norm_ker = math.sqrt(abs(np.sum(state.gradient_ker**2)))

			print state.likelihood, norm_ker, "threshold: ", state.paramKer[-1]

		diff = 1.

		cnt = 0.

		fig2 = plt.figure()
		fig2.show()
		ax5 = fig2.add_subplot(111)
		plt.ion()

		norm_nl = math.sqrt(abs(np.sum(state.gradient_NL**2)))

		while norm_nl > 0.1:

			v = np.arange(-50.,50.,1.)
			ax5.plot(v,fun.sigmoid(state.paramNL[0],v))
			fig2.canvas.draw()
			time.sleep(0.005)

			cnt = cnt + 1

			l0 = copy.copy(state.likelihood)

			state.iter_NL()
			state.update()

			norm_nl = abs(np.sum(state.gradient_NL**2))

			print "count NL: ",cnt

			print state.likelihood, norm_nl
	
	return state.paramNL,state.paramKer,state.likelihood

