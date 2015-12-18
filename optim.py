import numpy as np
import functions as fun
from scipy import signal
import copy
import diff_calc as diff
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
		self.membrane_potential = diff.MembPot(Model)

	def iter_ker(self,rate): 

		invH = np.linalg.pinv(self.hessian_ker)
		
		self.paramKer = self.paramKer - rate*np.dot(invH,self.gradient_ker)

	def iter_NL(self,rate):

		invH = np.linalg.pinv(self.hessian_NL)

		self.paramNL = self.paramNL - rate*np.dot(invH,self.gradient_NL)

	def update(self):

		self.sub_membrane_potential = diff.subMembPot(self)
		self.membrane_potential = diff.MembPot(self)

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

	iter_tot = 200.	

	while iter_tot > 2.:

		norm_ker = math.sqrt(abs(np.sum(state.gradient_ker**2)))

		diff_ker = 100.

		c0_ker = copy.copy(cnt_ker)

		while diff_ker > 3.:

			l0 = copy.copy(state.likelihood)

			print "count ker:", cnt_ker
			cnt_ker = cnt_ker + 1.

			ax1.plot(np.dot(state.paramKer[-10:-6],state.basisKer))
			ax2.plot(np.dot(state.paramKer[:4],state.basisKer))
			ax3.plot(np.dot(state.paramKer[-6:-1],state.basisASP))
			ax4.plot(state.membrane_potential[:3000])
			fig1.canvas.draw()			
			time.sleep(0.05)

			if cnt_ker<10:
				rate = 1.
			else:
				rate = 1.

			state.iter_ker(rate)
			state.update()

			ll = state.likelihood

			diff_ker = abs(np.around((ll - l0)*1000.))
				
			th = state.paramKer[-1]

			print "LL: ",ll,"diff: ",diff_ker,"thresh: ",th

		diff_nl = 100.

		c0_nl = copy.copy(cnt_nl)

		while diff_nl > 0.:

			NL = np.dot(state.paramNL,state.basisNL)

			ax5.plot(NL)
			fig2.canvas.draw()
			ax4.plot(state.membrane_potential[:3000])
			fig1.canvas.draw()
			time.sleep(0.005)

			cnt_nl = cnt_nl + 1.
			rate_nl = 1.

			print state.gradient_NL

			print state.hessian_NL	
			
			l0 = copy.copy(state.likelihood)

			state.iter_NL(rate_nl)
			state.update()

			diff_nl = abs(np.around((state.likelihood - l0)*10000.))

			print "count NL: ",cnt_nl

			print "ll: ",state.likelihood,"diff: ",diff_nl

		iter_tot = (cnt_nl-c0_nl) + (cnt_ker - c0_ker)	

		print "ITER_TOT: ", iter_tot
	
	return state.paramNL,state.paramKer,state.likelihood

