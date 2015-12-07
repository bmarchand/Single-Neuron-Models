import functions as fun
import numpy as np
import matplotlib.pylab as plt
import mechanisms as mech
import random
import math
import optim

class Synapses:

	tr_in = 3. #[ms] rise time inhibitory kernel
	td_in = 80. #[ms] decay time inhibitory kernel
	tr_exc = 1. #[ms] rise ' ' ' excitatory ' ' ' 
	td_exc = 20. #same
	N = 12 #number of pre-synaptic neurons

	def __init__(self,size,dt=0.025,len_ker=200.):

		self.dt = dt # timestep [ms]
		self.len_ker = len_ker #[ms] total-time over which the kernel is defined.
		self.len_ker_st = int(len_ker/dt) #like above, but in ms.
		self.ker = np.zeros((self.N,int(len_ker/dt)),dtype=float) #initialized at 0.
		l = [0,1,2,6,7,8,3,4,5,9,10,11] #shuffle indices to create groups.
		 
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
	compartments = 2 #number of subgroups. each of them has a non-linearity.
	non_linearity = [[100,1.,0.04],[100.,1.,0.04]] #params for NL (sigmoids)
	threshold = 30. #bio-inspired threshold value.
	PSP_size = 25. # mV: Roughly std of the memb. pot. in a compartment before NL 	  
	ASP_size = 20. # [mV] size of After-Spike-Potential (ASP)
	ASP_time = 200. #[ms] time-constant of decay ASP
	ASP_total = 1000. #[ms] total length of ASP
	ASP_total_st = ASP_total/dt #same but in time-steps unit.
	delta_v = 5.#[mv]thresh. selectivity. In LNLN-model, will be one. rest will adapt.
	lambda0 = 1. #[mV]firing value at threshold. 
	in_rate = 20. #with these parameters, in_rate = 20 -> ~8Hz output

class RunParameters:

	dt = 1. #[ms] same as above. dunno why defined twice (attribute access maybe).
	total_time = 60000. #total simulation time
	N = 12. #number of presynaptic neurons
	
class TwoLayerNeuron(Synapses,SpikingMechanism,RunParameters): 
# Neuron = synapses + spiking mechanisms + parameters.
	def __init__(self):

		self.synapses = Synapses(self.PSP_size,dt=self.dt) #instance of synapse class.
		self.add_input() #method defined below

	def add_input(self):
	
		self.input = [] #input will be a list of list of input spike times.
                        #one list per presynaptic neuron.
		for i in range(self.N):

			SpikeTrain = fun.spike_train(self) #self contains params. (e.g  in_rate)
			self.input.append(SpikeTrain) #append Poisson spiketrain with in_rate.

	def run(self): #generate output from input and spike mechanism.

		ctrl = control='off' #in case you wanna plot stuff.
		self.output,self.membrane_potential = mech.SpikeGeneration(self,ctrl)
		self.output_rate = len(self.output)/(0.001*self.total_time) #[Hz]

	def plot(self): #plot memb pot and its histogram.
	
		h = np.histogram(self.membrane_potential,bins=1000.)
		plt.plot(h[1][:-1],h[0])
		plt.show()
		absc = np.arange(int(self.total_time/self.dt)) #absc = abscisse (French)
		plt.plot(absc,self.membrane_potential)
		plt.show()

class FitParameters: # FitParameters is one component of the TwoLayerModel class

	dt = 1. #[ms]
	N = 12  # need to define it several times for access purposes.
	Ng = 2 # number of nsub_groups
	N_cos_bumps = 5 #number of PSP(ker) basis functions.
	len_cos_bumps = 500. #ms. total length of the basis functions 
	N_knots_ASP = 4.# number of knots for natural spline for ASP (unused)
	N_knots = 10. # number of knots for NL (unused)
	knots = range(-60,70,13) #knots for NL (unused)
	bnds = [-100.,100.] #[mV] domain over which NL is defined
	knots_ASP = range(int(100./dt),int(500./dt),100) #knots for ASP (unused)
	bnds_ASP = [0,500./dt] # domain over which ASP defined. [timesteps]
	basisNL = fun.Tents(knots,bnds,100000.) #basis for NL (Tents instead of splines)
	basisNLder = fun.DerTents(knots,bnds,100000.) #derivative of above. for gradient.
	basisNLSecDer = fun.SecDerTents(knots,bnds,100000.) #second derivative. 0 here.
	basisKer = fun.CosineBasis(N_cos_bumps,len_cos_bumps,dt) #basis for kernels.
	basisASP = fun.Tents(knots_ASP,bnds_ASP,1000.) #basis for ASP (Tents not splines)
	tol = 10**-6 #(Tol over gradient norm below which you stop optimizing)
 
class TwoLayerModel(FitParameters,RunParameters): #model object.

	def __init__(self): #initialized as a GLM (not working yet)

		dv = (self.bnds[1] - self.bnds[0])*0.00001 
		v = np.arange(self.bnds[0],self.bnds[1],dv)
		v = np.atleast_2d(v)

		para = fun.Ker2Param(v,self.basisNL)

		Ncosbumps = self.N_cos_bumps #just to make it shorter
		self.paramKer = np.zeros(int(self.N*Ncosbumps+self.N_knots_ASP+1.+1.)) 

	def add_data(self,neuron): #import data from neuron
		
		self.input = neuron.input
		self.output = [neuron.output]
		self.paramKer[-1] = -math.log(neuron.output_rate) #initialize for fit.

	def fit(self): #fit with block cooridinate ascend. not working yet.

		self.paramNL,self.paramKer,self.likelihood = optim.BlockCoordinateAscent(self)

	def plot(self): #under-developed plot method.

		print self.likelihood,self.paramKer
