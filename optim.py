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
		self.input_test = Model.input_test
		self.output_test = Model.output_test

		self.gradient_ker = diff.gradient_ker(Model) 
		self.gradient_NL = diff.gradient_NL(Model)

		self.hessian_ker = diff.hessian_ker(Model)
		self.hessian_NL = diff.hessian_NL(Model)

		self.paramKer = Model.paramKer
		self.paramNL = Model.paramNL

		self.lls = Model.lls
		self.Mds = Model.Mds
		self.switches = Model.switches
		#self.neustd = Model.neustd

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

		new_params = np.array([])

		old_b = copy.copy(self.basisNL)

		for g in range(self.Ng):
			
			Nbnl = old_b[g].shape[0]

			NL = np.dot(self.paramNL[g*Nbnl:(g+1)*Nbnl],self.basisNL[g]) # (1,100000)

			sptimes = np.floor(np.array(self.output)/self.dt) 
			sptimes = sptimes.astype('int')

			mp_sp_min = self.sub_membrane_potential[g,sptimes].min() #smallest mp spik
			mp_sp_max = self.sub_membrane_potential[g,sptimes].max() #highest mp spike

			new_bnds = [mp_sp_min,mp_sp_max]

			dv = (new_bnds[1] - new_bnds[0])*0.00001 #

			NL = diff.applyNL(NL,np.arange(new_bnds[0],new_bnds[1],dv),self.bnds[g]) #
			self.bnds[g] = new_bnds

			self.knots[g] = [mp_sp_min+((i+1)/10.)*(mp_sp_max-mp_sp_min) for i in range(9)]

			tots = 100000.

			self.basisNL[g] = fun.Tents(self.knots[g],self.bnds[g],tots)
			self.basisNLder[g] = fun.DerTents(self.knots[g],self.bnds[g],tots)
			self.basisNLSecDer[g] = fun.SecDerTents(self.knots[g],self.bnds[g],tots)

			dv = (self.bnds[g][1] - self.bnds[g][0])*0.00001 
			v = np.arange(self.bnds[g][0],self.bnds[g][1],dv)
			NL = np.atleast_2d(NL)
			para = fun.Ker2Param(NL,self.basisNL[g])
			new_params = np.hstack((new_params,para))	

		self.paramNL = new_params	
	
def BlockCoordinateAscent(Model):

	global state 
	state = State(Model)

	init_fun()
	
	while cnt > 4.:

		init_tot()
		
		try:
			while diff_ker > 4.:

				init_ker()
				state.iter_ker()
				state.update()

				finish_ker(10000.)
	
		except KeyboardInterrupt:
			pass			

		prepare_nl()

		try:
			while diff_nl > 2.:

				init_nl()		
				state.iter_NL()
				state.update()
				finish_nl(100000.)
	
		except KeyboardInterrupt:
			pass
		
	b = [state.basisNL,state.basisNLder,state.basisNLSecDer]
	sw = state.switches
	mds = state.Mds
			
	return state.paramNL,state.paramKer,state.lls,b,state.bnds,state.knots,mds,sw

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
	gs = gridspec.GridSpec(3,1)
	fig2 = plt.figure()
	fig2.show()
	ax5 = fig2.add_subplot(gs[0,0])
	ax6 = fig2.add_subplot(gs[1,0])
	ax62 = fig2.add_subplot(gs[2,0])
	plt.ion()	
	cnt = 40. #just need to be greater than 3.

	globals().update(locals())

def init_tot():

	global diff_ker,diff_nl,cnt

	diff_ker = 100.
	diff_nl = 100.

	cnt = 0.
	state.switches = state.switches + [state.likelihood]
	state.test()
	state.Mds = state.Mds + [state.Md]

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
	state.switches = state.switches + [state.likelihood]
	state.test()
	state.Mds = state.Mds + [state.Md]

	globals().update(locals())

def init_nl():

	ax4.plot(state.membrane_potential[:3000])
	fig1.canvas.draw()

	global l0
	l0 = copy.copy(state.likelihood)
			
	time.sleep(0.005)

	globals().update(locals())

def finish_nl(tol_fact):

	for g in range(state.Ng):
			
		Nbnl = state.basisNL[g].shape[0]
		NL = np.dot(state.paramNL[Nbnl*g:(g+1)*Nbnl],state.basisNL[g]) # (1,100000)

		std = state.sub_membrane_potential[g,:].std()
		dv = (state.bnds[g][1]-state.bnds[g][0])*0.00001
		V = np.arange(state.bnds[g][0],state.bnds[g][1],dv)			
		NL = np.dot(state.paramNL[g*Nbnl:(g+1)*Nbnl],state.basisNL[g])
		#Y = fun.sigmoid([0.,25.,1.,1./25.],(state.neustd/std)*V)/2.
	
		
		if g==0:
			#ax5.plot(V,Y)
			ax5.plot(V,NL)
		elif g==1:
			#ax6.plot(V,Y)
			ax6.plot(V,NL)
		elif g==2:
			ax62.plot(V,NL)

	fig2.canvas.draw()

	global diff_nl
	diff_nl = abs(np.floor((state.likelihood - l0)*tol_fact))
	state.lls = state.lls + [state.likelihood]

	print "(NL)","LL: ",state.likelihood,"diff: ",diff_nl
	global cnt
	cnt = cnt + 1.
	globals().update(locals())

