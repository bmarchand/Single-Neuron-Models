import functions as fun
import numpy as np
import matplotlib.pylab as plt
import mechanisms as mech
import random
import math
import optim
import diff_calc as diff
from matplotlib import gridspec

class Synapses:

	tr_in = 3. #[ms] rise time inhibitory kernel
	td_in = 80. #[ms] decay time inhibitory kernel
	tr_exc = 1. #[ms] rise ' ' ' excitatory ' ' ' 
	td_exc = 30. #same
	N =  12#number of pre-synaptic neurons

	def __init__(self,size,dt=1.,len_ker=300.):

		self.dt = dt # timestep [ms]
		self.len_ker = len_ker #[ms] total-time over which the kernel is defined.
		self.len_ker_st = int(len_ker/dt) #like above, but in ms.
		self.ker = np.zeros((self.N,int(len_ker/dt)),dtype=float) #initialized at 0.
		l = [6,7,8,0,1,2,3,4,5,9,10,11]		
		#l = [12,13,14,15,16,17,18,19,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11] #shuffle indices to create groups.
		 
		for i in range(self.N): #loop over the presynaptic neurons

			if i<(self.N/2): #first group (inh. and exc. thks to shuffling)

				trexc = fun.jitter(self.tr_exc) #tr_exc + noise
				tdexc = fun.jitter(self.td_exc) #same
				exc_ker = fun.alpha_fun(trexc,tdexc,self.dt,self.len_ker) #alpha_fun
				self.ker[l[i],:] = fun.jitter(size)*exc_ker #size is added.

			else:

				trin = fun.jitter(self.tr_in) #same.
				tdin = fun.jitter(self.td_in)
				inh_ker = fun.alpha_fun(trin,tdin,self.dt,self.len_ker)		
				self.ker[l[i],:] = -fun.jitter(size)*inh_ker

	def plot(self): #plotting the kernels defined above.

		for i in range(self.N):

			plt.plot(self.ker[i,:])
		
		plt.show()

class SpikingMechanism: #gather parameters for spiking.

	dt = 1. #[ms]
	strh = 25.
	Ng = 2 #number of subgroups. each of them has a non-linearity.
	non_linearity = [[0.,strh,1.,1/strh],[0.,strh,1.,1/strh]]
	threshold = 30. #bio-inspired threshold value.
	PSP_size = 25.# mV: Roughly std of the memb. pot. in a compartment before NL 	  
	ASP_size = 10. # [mV] size of After-Spike-Potential (ASP)
	ASP_time = 200. #[ms] time-constant of decay ASP
	ASP_total = 1000. #[ms] total length of ASP
	ASP_total_st = ASP_total/dt #same but in time-steps unit.
	delta_v = 1. #[mv]thresh. selectivity. In LNLN-model, will be one. rest will ada
	lambda0 = 1. #[mV]firing value at threshold. 
	in_rate = 25. #with these parameters, in_rate = 20 -> ~8Hz output

class RunParameters:

	total_time = 600000. #total simulation time
	total_time_test = 10000.
	N = 12 #number of presynaptic neurons
	
class TwoLayerNeuron(Synapses,SpikingMechanism,RunParameters): 
# Neuron = synapses + spiking mechanisms + parameters.
	def __init__(self):

		self.synapses = Synapses(self.PSP_size,dt=self.dt) #instance of synapse class.

	def add_input(self):
	
		self.input = [] #input will be a list of list of input spike times.
                        #one list per presynaptic neuron.
		self.input_test = []

		for i in range(self.N):

			SpikeTrain = fun.spike_train(self,'training') #self contains params. (e.g  in_rate)
			SpikeTrain_test = fun.spike_train(self,'test')

			self.input.append(SpikeTrain) #append Poisson spiketrain with in_rate.
			self.input_test.append(SpikeTrain_test)			

	def run(self): #generate output from input and spike mechanism.

		ctrl = control = 'off' #in case you wanna plot stuff.
		out,v,sub_memb = mech.SpikeGeneration(self.input,self,ctrl,'training')

		self.output = out
		self.membrane_potential = v
		self.sub_memb_pot = sub_memb

		self.output_rate = len(self.output)/(0.001*self.total_time) #[Hz]

		self.output_test = []

		for i in range(100):

			out_test,v_test,s = mech.SpikeGeneration(self.input_test,self,ctrl,'test')

			self.output_test = self.output_test + [out_test]
		
	def plot(self): #plot memb pot and its histogram.
	
		h = np.histogram(self.membrane_potential,bins=1000.)
		plt.plot(h[1][:-1],h[0])
		plt.show()
		absc = np.arange(int(self.total_time/self.dt)) #absc = abscisse (French)
		plt.plot(absc,self.membrane_potential)
		plt.show()

class FitParameters(): # FitParameters is one component of the TwoLayerModel class

	def __init__(self,basis='Tents'):

		self.basis_str = basis

	dt = 1. #[ms]
	N = 12 # need to define it several times for access purposes.
	Ng = 2 # number of nsub_groups
	compartment = 2
	N_cos_bumps = 7 #number of PSP(ker) basis functions.
	len_cos_bumps = 300. #ms. total length of the basis functions 
	N_knots_ASP = 5.# number of knots for natural spline for ASP (unused)

	knots = []
	bnds = []

	for i in range(Ng):

		knots = knots + [range(-50,60,10)]
		bnds = bnds + [[-100.,100.]]

	knots_ASP = range(int(100./dt),int(600./dt),int(100/dt) ) #knots for ASP (unused)
	bnds_ASP = [0,600./dt] # domain over which ASP defined. [timesteps]

	basisNL = []
	basisNLder = []
	basisNLSecDer = []

	for i in range(Ng):

		basisNL = basisNL + [fun.Tents(knots[i],bnds[i],100000.)] #basis for NL 
		basisNLder = basisNLder + [fun.DerTents(knots[i],bnds[i],100000.)] 
		basisNLSecDer = basisNLSecDer + [fun.SecDerTents(knots[i],bnds[i],100000.)] 

	knots_ker = [2./dt,5./dt,10./dt,20./dt,40./dt,100./dt,200./dt]
	#basisKer = fun.NaturalSpline(knots_ker,[0.,len_cos_bumps/dt],len_cos_bumps/dt)[1:
	basisKer = fun.CosineBasis(N_cos_bumps,len_cos_bumps,dt,a=2.0)
	basisASP = fun.Tents(knots_ASP,bnds_ASP,600.) #basis for ASP (Tents not splines)
	tol = 10**-6 #(Tol over gradient norm below which you stop optimizing)

	def plot(self):

		for i in range(np.shape(self.basisKer)[0]):
			plt.plot(self.basisKer[i,:])
		plt.show()
 
class TwoLayerModel(FitParameters,RunParameters): #model object.

	def __init__(self): #initialized as a GLM (not working yet)

		ps = []

		self.paramNL = np.array([])

		for i in range(self.Ng):

			dv = (self.bnds[i][1] - self.bnds[i][0])*0.00001 
			v = np.arange(self.bnds[i][0],self.bnds[i][1],dv)
			v = np.atleast_2d(v)

			self.paramNL = np.hstack((self.paramNL,fun.Ker2Param(v,self.basisNL[i])))

		self.lls = []
		self.switches = []
		self.Mds = []
		
		Nb = self.basisKer.shape[0]     #just to make it shorter

		self.paramKer = np.zeros(int(self.N*Nb+self.basisASP.shape[0]+1)) 

	def add_data(self,neuron): #import data from neuron

		Nsteps = neuron.total_time/self.dt
		
		self.input = neuron.input
		self.output = [neuron.output]
		self.paramKer[-1] = -math.log(len(neuron.output)/Nsteps) #initialize for fit.

		self.neustd = neuron.sub_memb_pot.std()

		self.sub_membrane_potential = diff.subMembPot(self,'training')
		self.membrane_potential = diff.MembPot(self)

		self.input_test = neuron.input_test
		self.output_test = neuron.output_test

	def normalize_basis(self):

		for g in range(Ng):
			for i in range(self.basisNL.shape[0]):

				sm = np.sum(self.basisNL[g][i,:])

				self.basisNLder[g][i,:] = self.basisNLder[g][i,:]/sm
				self.basisNLSecDer[g][i,:] = self.basisNLSecDer[g][i,:]/sm
				self.basisNL[g][i,:] = self.basisNL[g][i,:]/sm

	def membpot(self):

		self.sub_membrane_potential = diff.subMembPot(self,'training')
		self.membrane_potential = diff.MembPot(self)

	def fit(self): #fit with block cooridinate ascend. not working yet.

		pNL,pKr,lls,b,bn,k,mds,sw = optim.BlockCoordinateAscent(self)

		self.lls = lls
		self.basisNL = b[0]
		self.basisNLder = b[1]
		self.basisNLSecDer = b[2]
		self.bnds = bn
		self.knots = k
		self.Mds = mds
		self.switches = sw
		self.paramKer = pKr
		self.paramNL = pNL

	def test(self):

		Nb = self.basisKer.shape[0]

		self.out_model_test = []
		self.sub_memb_pot_test = diff.subMembPot(self,'test')

		print "testing ..."

		for i in range(100):
	
			self.out_model_test = self.out_model_test + [mech.run_model(self.input_test,self)]

		self.delta_md = 4.

		self.Md = fun.SimMeas(self.out_model_test,self.output_test,self,self.delta_md)

	def plot(self): #under-developed plot method.

		for g in range(self.Ng):

			dv = (self.bnds[g][1]-self.bnds[g][0])*0.00001
			V = np.arange(self.bnds[g][0],self.bnds[g][1],dv)
			mostd = self.sub_membrane_potential[g,:].std()
			neustd = neuron.sub_memb_pot.std()
			Y = fun.sigmoid([0.,25.,1.,1./25.],((neustd/mostd)*V)/neuron.delta_v)

			fig5 = plt.figure()

			gs = gridspec.GridSpec(self.Ng,1)

			Nbnl = self.basisNL[g].shape[0]
			NL = np.dot(self.paramNL[g*Nbnl:(g+1)*Nbnl],self.basisNL[g])

			ax = fig5.add_subplot(gs[g,0])

			ax.plot(V,Y-Y.mean()+NL.mean())
			ax.plot(V,NL)
			ax.set_xlabel("membrane potential ('mV')")
			ax.set_ylabel("membrane potential, after non-linearity ('mV')") 

		fig5.show()
		fig6 = plt.figure()
		axker = fig6.add_subplot(111)
		Ker = np.zeros((neuron.N,self.len_cos_bumps/self.dt),dtype='float')
		Nb = self.basisKer.shape[0]
		Ker[-1,:] = np.dot(self.paramKer[:Nb],self.basisKer)
		Ker[0,:] = np.dot(self.paramKer[(self.N-1)*Nb:self.N*Nb],self.basisKer)
		std = neuron.sub_memb_pot.std()
		axker.plot(Ker[-1,:])
		axker.plot(Ker[0,:])
		subsa = (self.dt/neuron.dt)
		axker.plot(neuron.PSP_size*mostd*neuron.synapses.ker[-1,::subsa]/std)
		axker.plot(neuron.PSP_size*mostd*neuron.synapses.ker[0,::subsa]/std)
		axker.set_xlabel("time (ms)")
		axker.set_ylabel("PSP ('mV')")
		fig6.show()
		fig7 = plt.figure()
		axlls = fig7.add_subplot(111) 
		axlls.plot(self.lls,'bo')
		fig7.show()

		fig8 = plt.figure()
		axmd = fig8.add_subplot(gs[0,0])
		axmd.plot(self.Mds,'bo')
		axmd.set_xlabel("Block Optimization number")
		axmd.set_ylabel("Md - percentage of predicted spikes")
		
		axsw = fig8.add_subplot(gs[1,0])
		axsw.plot(self.switches,'bo')
		axsw.set_xlabel("Block Optimization number")
		axsw.set_ylabel("Log-likelihood (bits/spike)")
		fig8.show()

	def save(self):

		np.savetxt('paramnl.txt',self.paramNL)
		np.savetxt('paramker.txt',self.paramKer)
		np.savetxt('knots.txt',self.knots)
		np.savetxt('bnds.txt',self.bnds)





		

