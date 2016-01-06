import functions as fun
from scipy import signal
import copy
import diff_calc as diff
import main
import matplotlib.pylab as plt
from matplotlib import gridspec
import math
import time
import numpy as np

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

		invH = np.linalg.inv(self.hessian_ker)
		
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

	def initialize_ker(self):

		self.paramKer = np.zeros(self.paramKer.shape)

		Nsteps = self.total_time/self.dt

		self.paramKer[-1] = - math.log(len(self.output[0])/Nsteps)

		self.update()

	def renormalize(self):

		std = self.sub_membrane_potential.std()
		
		self.paramKer = (1./std)*self.paramKer

		self.membrane_potential = diff.MembPot(self)

		self.update()	

	def renorm_basis(self):

		NL = np.dot(self.paramNL,self.basisNL)

		std = self.sub_membrane_potential.std()
		mu = self.sub_membrane_potential.mean()

		new_knots = [mu-2*std,mu-std,mu,mu+std,mu+2*std]

		self.basisNL = fun.NaturalSpline(new_knots,self.bnds,100000.)
		self.basisNLder = fun.DerNaturalSpline(new_knots,self.bnds,100000.)
		self.basisNLSecDer = fun.SecDerNaturalSpline(new_knots,self.bnds,100000.)

		self.normalize_basis()

		
def BlockCoordinateAscent(Model):

	params_nl = []
	params_ker = []

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

	fig3 = plt.figure()
	fig3.show()
	ax6 = fig3.add_subplot(111)
	plt.ion()	

	iter_tot = 200.	

	params_nl = params_nl + [state.paramNL]
	params_ker = params_ker + [state.paramKer]

	while iter_tot > 3.:

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

			h_sub = np.histogram(state.sub_membrane_potential,bins = 1000,range=[-80.,80.])
			h_aft = np.histogram(state.membrane_potential,bins = 1000,range=[-80,80])
			
			ax6.plot(h_sub[1][:-1],h_sub[0])
			ax6.plot(h_aft[1][:-1],h_aft[0])

			fig3.canvas.draw()
			
			time.sleep(0.05)

			if cnt_ker<10:
				rate = 1.
			else:
				rate = 1.

			state.iter_ker(rate)
			state.update()

			params_ker = params_ker + [state.paramKer]

			ll = state.likelihood

			diff_ker = abs(np.around((ll - l0)*1000.))
				
			th = state.paramKer[-1]

			print "LL: ",ll,"diff: ",diff_ker,"thresh: ",th

		diff_nl = 100.

		state.renormalize()

		c0_nl = copy.copy(cnt_nl)

		V = np.arange(state.bnds[0],state.bnds[1],(state.bnds[1]-state.bnds[0])*0.00001)

		Y = fun.sigmoid([-25.,50.,1.,1./25.],37.69*V)/5.
		ax5.plot(V,Y)

		while diff_nl > 0.:

			V = np.arange(state.bnds[0],state.bnds[1],(state.bnds[1]-state.bnds[0])*0.00001)
			NL = np.dot(state.paramNL,state.basisNL)

			ax5.plot(V,NL)
			fig2.canvas.draw()
			ax4.plot(state.membrane_potential[:3000])
			fig1.canvas.draw()

			h_sub = np.histogram(state.sub_membrane_potential,bins = 1000,range=[-80.,80.])
			h_aft = np.histogram(state.membrane_potential,bins = 1000,range=[-80,80])
			
			ax6.plot(h_sub[1][:-1],h_sub[0])
			ax6.plot(h_aft[1][:-1],h_aft[0])
			ax6.set_ylim([0,1000.])

			fig3.canvas.draw()
			
			time.sleep(0.005)

			cnt_nl = cnt_nl + 1.
			rate_nl = 1.
			
			l0 = copy.copy(state.likelihood)

			state.iter_NL(rate_nl)
			state.update()

			params_nl = params_nl + [state.paramNL]

			diff_nl = abs(np.around((state.likelihood - l0)*10000.))

			print "count NL: ",cnt_nl

			print "ll: ",state.likelihood,"diff: ",diff_nl

		iter_tot = (cnt_nl-c0_nl) + (cnt_ker - c0_ker)	

		print "ITER_TOT: ", iter_tot

		params_nl = np.array(params_nl)
		params_ker = np.array(params_ker)

		np.savetxt('params_nl.txt',params_nl)
		np.savetxt('params_ker.txt',params_ker)		
		
	return state.paramNL,state.paramKer,state.likelihood

