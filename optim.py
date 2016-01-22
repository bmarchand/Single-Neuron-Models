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

		self.lls = Model.lls

		self.likelihood = diff.likelihood(Model)

		self.sub_membrane_potential = diff.subMembPot(Model,'training')
		self.membrane_potential = diff.MembPot(Model)

	def iter_ker(self): 

		invH = np.linalg.inv(self.hessian_ker)
		
		self.paramKer = self.paramKer - np.dot(invH,self.gradient_ker)

	def iter_NL(self):

		invH = np.linalg.pinv(self.hessian_NL)

		self.paramNL = self.paramNL - np.dot(invH,self.gradient_NL)

	def update(self):

		self.sub_membrane_potential = diff.subMembPot(self,'training')
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

		NL = np.dot(self.paramNL,self.basisNL) # (1,100000)

		sptimes = np.floor(np.array(self.output)/self.dt) 
		sptimes = sptimes.astype('int')

		mp_sp_min = self.sub_membrane_potential[:,sptimes].min() #smallest mp spike
		mp_sp_max = self.sub_membrane_potential[:,sptimes].max() #highest mp spike

		print "min_memb_pot: ",mp_sp_min,mp_sp_max

		mu = self.sub_membrane_potential[:,sptimes].mean()
		std = self.sub_membrane_potential[:,sptimes].std()

		new_bnds = [mp_sp_min,mp_sp_max]

		dv = (new_bnds[1] - new_bnds[0])*0.00001 #

		NL = diff.applyNL(NL,np.arange(new_bnds[0],new_bnds[1],dv),self) # need to have a vector with the values of the NL on the new axis, so that later, can get the parameters that correspond in the new basis.

		self.bnds = new_bnds

		dok = (mp_sp_max - mp_sp_min)*0.05

		self.knots = [kn for kn in np.arange(mp_sp_min+dok,mp_sp_max-dok,dok)]

		self.basisNL = fun.Tents(self.knots,self.bnds,100000.)
		self.basisNLder = fun.DerTents(self.knots,self.bnds,100000.)
		self.basisNLSecDer = fun.SecDerTents(self.knots,self.bnds,100000.)

		dv = (self.bnds[1] - self.bnds[0])*0.00001 
		v = np.arange(self.bnds[0],self.bnds[1],dv)
		NL = np.atleast_2d(NL)
		para = fun.Ker2Param(NL,self.basisNL)
		self.paramNL = para		
	
def BlockCoordinateAscent(Model):

	global state 
	state = State(Model)

	init_fun()
	
	while cnt > 5.:

		init_tot()
		
		while diff_ker > 3.:

			init_ker()
			state.iter_ker()
			state.update()

			finish_ker(1000.)
			
		prepare_nl()

		while diff_nl > 2.:

			init_nl()		
			state.iter_NL()
			state.update()
			finish_nl(1000000.)
			
	return state.paramNL,state.paramKer,state.lls,state.basisNL,state.bnds

#Below: Superficial layer of code for plotting, counting iterations and such.

def init_fun():

	gs = gridspec.GridSpec(4,1)
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(gs[0,0])
	ax2 = fig1.add_subplot(gs[1,0])
	ax3 = fig1.add_subplot(gs[2,0])
	ax4 = fig1.add_subplot(gs[3,0])
	fig1.show()
	plt.draw()
	plt.ion()
	gs = gridspec.GridSpec(2,1)
	fig2 = plt.figure()
	fig2.show()
	ax5 = fig2.add_subplot(gs[0,0])
	plt.ion()	
	cnt = 40. #just need to be greater than 3.

	globals().update(locals())

def init_tot():

	global diff_ker,diff_nl,cnt

	diff_ker = 100.
	diff_nl = 100.

	cnt = 0.

def init_ker():

	nk = state.basisKer.shape[0]
	na = state.basisASP.shape[0]+1

	ax1.plot(np.dot(state.paramKer[-na-nk:-na],state.basisKer))
	ax2.plot(np.dot(state.paramKer[:nk],state.basisKer))
	ax3.plot(np.dot(state.paramKer[-na:-1],state.basisASP))
	ax4.plot(state.membrane_potential[:3000])
	fig1.canvas.draw()			
			
	time.sleep(0.05)

	l0 = copy.copy(state.likelihood)

	global cnt
	cnt = cnt + 1

	globals().update(locals())

def finish_ker(tol_fact):

	ll = state.likelihood

	diff_ker = abs(np.floor((ll - l0)*tol_fact))
				
	th = state.paramKer[-1]

	state.lls = state.lls + [ll]

	print "(ker)","LL: ",ll,"diff: ",diff_ker,"thresh: ",th

	globals().update(locals())

def prepare_nl():

	state.renorm_basis()
	state.update()

	globals().update(locals())

def init_nl():

	std = state.sub_membrane_potential.std()
	dv = (state.bnds[1]-state.bnds[0])*0.00001
	V = np.arange(state.bnds[0],state.bnds[1],dv)			
	NL = np.dot(state.paramNL,state.basisNL)
	Y = fun.sigmoid([0.,25.,1.,1./25.],(36./std)*V)/2.
	ax5.plot(V,Y)
	ax5.plot(V,NL)

	fig2.canvas.draw()
	ax4.plot(state.membrane_potential[:3000])
	fig1.canvas.draw()

	global l0
	l0 = copy.copy(state.likelihood)
			
	time.sleep(0.005)

	globals().update(locals())

def finish_nl(tol_fact):

	global diff_nl
	diff_nl = abs(np.floor((state.likelihood - l0)*tol_fact))
	state.lls = state.lls + [state.likelihood]

	print "(NL)","LL: ",state.likelihood,"diff: ",diff_nl
	global cnt
	cnt = cnt + 1.
	globals().update(locals())

