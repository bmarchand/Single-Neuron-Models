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

	def iter_NL(self,rate):

		invH = np.linalg.pinv(self.hessian_NL)

		for g in range(len(self.paramNL)):

			Np = len(self.paramNL[g])	
			Ns = len(self.output[0])

			invh = 1./self.hessian_NL[0,0]

			self.paramNL[g][0] = self.paramNL[g][0] - invh*self.gradient_NL[g*Np]  

			invh = 1./self.hessian_NL[1,1]

			self.paramNL[g][1] = self.paramNL[g][1] - invh*self.gradient_NL[g*Np+1]

			step = 0.005*self.gradient_NL[g*Np+2]/self.gradient_NL[g*Np+2]

			self.paramNL[g][2] = self.paramNL[g][2] + step		

			step = 0.005*self.gradient_NL[g*Np+3]/self.gradient_NL[g*Np+3]
	
			self.paramNL[g][3] = self.paramNL[g][3] + step
	
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

	cnt_ker = 1.
	cnt_nl = 1.

	gs = gridspec.GridSpec(4,1)

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(gs[0,0])
	ax2 = fig1.add_subplot(gs[1,0])
	ax3 = fig1.add_subplot(gs[2,0])
	ax4 = fig1.add_subplot(gs[3,0])

	fig1.show()
	plt.draw()
	plt.ion()

	
	fig2 = plt.figure()
	fig2.show()
	ax5 = fig2.add_subplot(111)
	plt.ion()

	while norm>Model.tol:

		norm_ker = math.sqrt(abs(np.sum(state.gradient_ker**2)))

		tol_ker = copy.copy(12./cnt_ker)

		while norm_ker > tol_ker:

			l0 = copy.copy(state.likelihood)

			print "count ker:", cnt_ker
			cnt_ker = cnt_ker + 1.		

			ax1.plot(np.dot(state.paramKer[-10:-6],state.basisKer))
			ax2.plot(np.dot(state.paramKer[:4],state.basisKer))	
			ax3.plot(np.dot(state.paramKer[-6:-1],state.basisASP))		
			ax4.plot(1-np.exp(-np.exp(state.membrane_potential[:3000])))
			fig1.canvas.draw()	
			time.sleep(0.005)

			if cnt_ker<10:
				rate = 1.
			else:
				rate = 1.

			state.iter_ker(rate)
			state.update()
			
			norm_ker = math.sqrt(abs(np.sum(state.gradient_ker**2)))

			ll = state.likelihood
			th = state.paramKer[-1]

			print "LL: ",ll, "norm: ",norm_ker, "thresh: ",th,"tol: ", tol_ker

		tol_nl = copy.copy(0.005/cnt_nl)

		norm_nl = math.sqrt(abs(np.sum(state.gradient_NL**2)))

		while norm_nl > tol_nl:

			v = np.arange(-50.,50.,1.)
			ax5.plot(v,fun.sigmoid(state.paramNL[0],v))
			fig2.canvas.draw()
			ax4.plot(1-np.exp(-np.exp(state.membrane_potential[:3000])))
			fig1.canvas.draw()	
			time.sleep(0.005)

			cnt_nl = cnt_nl + 1.
			rate_nl = 0.001/cnt_nl

			l0 = copy.copy(state.likelihood)

			state.iter_NL(rate_nl)
			state.update()

			norm_nl = abs(np.sum(state.gradient_NL**2))

			print "count NL: ",cnt_nl

			print state.likelihood, norm_nl, "tol: ",tol_nl
	
	return state.paramNL,state.paramKer,state.likelihood

