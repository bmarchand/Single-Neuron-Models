from __future__ import division
import functions as fun #import useful functions, like sigmoids, Md-calculations
import numpy as np
import matplotlib.pylab as plt
import mechanisms as mech #contains code to run LNLN model and produce spikes
import random
import math
import optim # contains code for fitting LNLN on spiking data
import diff_calc as diff
from matplotlib import gridspec
import interface # contains code for importing data produced by NEURON
import pickle

dataset = pickle.load(open("dataset2.p","rb")) #importing dataset2 specificities


class Synapses: #Used to produce artificial data.

	tr_in = 3. #[ms] rise time inhibitory kernel
	td_in = 80. #[ms] decay time inhibitory kernel
	tr_exc = 1. #[ms] rise ' ' ' excitatory ' ' ' 
	td_exc = 30. #same
	N =  18#number of pre-synaptic neurons

	def __init__(self,size,dt=1.,len_ker=300.):

		self.dt = dt # timestep [ms]
		self.len_ker = len_ker #[ms] total-time over which the kernel is defined.
		self.len_ker_st = int(len_ker/dt) #like above, but in ms.
		self.ker = np.zeros((self.N,int(len_ker/dt)),dtype=float) #initialized at 0.
		l = [0,1,2,6,7,8,12,13,14,3,4,5,9,10,11,15,16,17] #exc. syn. go in the first half. Just a way to label synapses as exc or inh and put them into groups.

		for i in range(self.N): #loop over the presynaptic neurons

			if i<(self.N/2): #excitatory synapses

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

class SpikingMechanism: #various parameters used for producing artificial data.

	dt = 1. #[ms]
	Ng = 3 #number of subgroups. each of them has a non-linearity.
	non_linearity = [[1.,0.02,0.01],[0.5,0.,0.02],[1.,0.05,0.03]] #parameters of NLs
	threshold = 2. #bio-inspired threshold value.
	PSP_size = 1.# mV: Roughly std of the memb. pot. in a compartment before NL 	  
	ASP_size = 0.8 # [mV] size of After-Spike-Potential (ASP)
	ASP_time = 20. #[ms] time-constant of decay ASP
	ASP_total = 100. #[ms] total length of ASP
	ASP_total_st = ASP_total/dt #same but in time-steps unit.
	delta_v = 0.02 #[mv]thresh. selectivity. In LNLN-model, will be one. rest will ada
	lambda0 = 1. #[mV]firing value at threshold. 

class RunParameters:

	total_time = 1200000.#dataset["total_time"] #total simulation time
	total_time_test = 40000. #dataset["total_time_test"]
	N = 18 # dataset["N"] #number of presynaptic neurons
	
class TwoLayerNeuron(Synapses,SpikingMechanism,RunParameters): 
# Neuron = synapses + spiking mechanisms + parameters.
	def __init__(self):

		self.synapses = Synapses(self.PSP_size,dt=self.dt) #instance of synapse class.

	def add_input(self):
	
		self.input = [] #input will be a list of list of input spike times.
                        #one list per presynaptic neuron.
		self.input_test = []

		self.windows = {}

		self.windows['training'	] = [[0.,200000.,10.,10.,10.,10.,10.,10.],
									[200000.,400000.,10.,10.,10.,10.,60.,10.],
									[400000.,600000.,10.,10.,60.,10.,10.,10.],
									[600000.,800000.,60.,10.,10.,10.,10.,10.],
									[800000.,1000000.,40.,10.,40.,10.,10.,10.],
									[1000000.,1200000.,25.,10.,25.,10.,25.,10.]]

		# convention: [[tstart window1,tstop window1,ex ],[],]
		
		self.windows['test'] = [[0.,10000.,10.,10.,10.,10.,10.,60.],
									[10000.,20000.,60.,10.,10.,10.,10.,10.],
									[20000.,30000.,40.,10.,40.,10.,10.,10.],
									[30000.,40000.,25.,10.,25.,10.,25.,10.]]

		self.l = [0,1,2,6,7,8,12,13,14,3,4,5,9,10,11,15,16,17]

		for i in range(self.N):

			gr = int(i/(self.N/self.Ng))

			if i%((self.N/self.Ng))<(self.N/self.Ng)/2:

				SpikeTrain = fun.spike_train(self,gr,'exc','training') 
				SpikeTrain_test = fun.spike_train(self,gr,'exc','test')
				self.input.append(SpikeTrain) 
				self.input_test.append(SpikeTrain_test)		

			else:

				SpikeTrain = fun.spike_train(self,gr,'inh','training') 
				SpikeTrain_test = fun.spike_train(self,gr,'inh','test')
				self.input.append(SpikeTrain) 
				self.input_test.append(SpikeTrain_test)		

		self.input = self.input[::-1]
		self.input_test = self.input_test[::-1]


	def run(self): #generate output from input and spike mechanism.

		ctrl = control = 'off' #in case you wanna plot stuff.
		out,v,sub_memb = mech.SpikeGeneration(self.input,self,ctrl,'training')

		self.output = out

		rates = []

		for i in range(20):

			cnt = 0.

			for st in out:

				if (st>((i/20.)*self.total_time))&(st<(((i+1)/20.)*self.total_time)):

					cnt = cnt + 1.

			rates = rates + [cnt/(0.001*0.05*self.total_time)]

		figrates = plt.figure()

		axrate = figrates.add_subplot(111)

		axrate.plot(rates)
	
		figrates.show()				
	
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
		plt.plot(self.membrane_potential)
		plt.show()

class BBPneuron(RunParameters):

	def __init__(self):

		path = dataset["path"]

		strs = dataset["strs"]

		grps = dataset["grps"]

		inp_test,inp,outtest,out = interface.import_data([0,1],path,strs,grps)

		outtest = np.array(outtest)

		outtest = outtest[outtest<self.total_time_test]

		out = np.array(out)

		out = out[out<self.total_time]

		for i in range(len(inp_test)):

			inter = np.array(inp_test[i])
			inp_test[i] = list(inter[inter<self.total_time_test])

			inp_tmp = np.array(inp[i])

			inp[i] = list(inp_tmp[inp_tmp<self.total_time])

		self.output_test = [list(outtest)]
		self.output = list(out)
		self.input = inp
		self.input_test = inp_test


class FitParameters(): # FitParameters is one component of the TwoLayerModel class

	def __init__(self,basis='Tents'):

		self.basis_str = basis

	dt = 1.  #[ms]
	N = 18#dataset["N"]# need to define it several times for access purposes.
	Ng = 3#dataset["Ng"] # number of nsub_groups
	Nneur = [range(0,7),range(6,13),range(12,19)]#dataset["Nneur"]
	N_cos_bumps = 5 #number of PSP(ker) basis functions.
	len_cos_bumps = 300. #ms. total length of the basis functions 
	N_knots_ASP = 5.# number of knots for natural spline for ASP (unused)

	knots = []
	bnds = []

	for i in range(Ng):

		knots = knots + [range(-50,60,10)]
		bnds = bnds + [[-100.,100.]]
	knots_ASP = range(int(15/dt),int(60./dt),int(10/dt) ) #knots for ASP (unused)
	bnds_ASP = [0,60./dt] # domain over which ASP defined. [timesteps]

	basisNL = []
	basisNLder = []
	basisNLSecDer = []

	knots_back_prop = [10./dt,30./dt,70./dt,150./dt]

	#basisBackProp = fun.Cst(knots_back_prop,[0,len_cos_bumps/dt],len_cos_bumps/dt)

	flag = ['nope','nope','nope']

	for i in range(Ng):

		basisNL = basisNL + [fun.Tents(knots[i],bnds[i],100000.)] #basis for NL 
		basisNLder = basisNLder + [fun.DerTents(knots[i],bnds[i],100000.)] 
		basisNLSecDer = basisNLSecDer + [fun.SecDerTents(knots[i],bnds[i],100000.)] 

	knots_ker = [2./dt,5./dt,10./dt,20./dt,30./dt,80./dt,100./dt]

	for i in range(len(knots_ker)):

		knots_ker[i] = len_cos_bumps - knots_ker[i]

	#basisKer = fun.NaturalSpline(knots_ker,[0.,len_cos_bumps/dt],len_cos_bumps/dt)
	#basisKer = basisKer[1:,::-1]
	basisKer = fun.CosineBasis(N_cos_bumps,len_cos_bumps,dt,a=1.7)
	basisKer = basisKer[1:,:]
	basisASP = fun.Tents(knots_ASP,bnds_ASP,60.) #basis for ASP (Tents not splines)
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
		#Nbbp = self.basisBackProp.shape[0]

		self.paramKer = np.zeros(int(self.N*Nb+self.basisASP.shape[0]+1)) 

	def add_data(self,neuron): #import data from neuron

		Nsteps = neuron.total_time/self.dt
		
		self.input = neuron.input
		self.output = [neuron.output]

		self.paramKer[-1] = -math.log(len(neuron.output)/float(neuron.total_time)) #initialize for fit.

		#self.neustd = neuron.sub_memb_pot.std()

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
			#neustd = neuron.sub_memb_pot.std()
			#Y = fun.sigmoid([0.,25.,1.,1./25.],((neustd/mostd)*V)/neuron.delta_v)

			fig5 = plt.figure()

			gs = gridspec.GridSpec(self.Ng,1)

			Nbnl = self.basisNL[g].shape[0]
			NL = np.dot(self.paramNL[g*Nbnl:(g+1)*Nbnl],self.basisNL[g])

			ax = fig5.add_subplot(gs[g,0])

			#ax.plot(V,Y-Y.mean()+NL.mean())
			ax.plot(V,NL)
			ax.set_xlabel("membrane potential ('mV')")
			ax.set_ylabel("membrane potential, after non-linearity ('mV')") 

		fig5.show()
		fig6 = plt.figure()
		axker = fig6.add_subplot(111)
		Ker = np.zeros((self.N,self.len_cos_bumps/self.dt),dtype='float')
		Nb = self.basisKer.shape[0]

		for i in range(self.N):

			Ker[i,:] = np.dot(self.paramKer[i*Nb:(i+1)*Nb],self.basisKer)
	
		#std = neuron.sub_memb_pot.std()

		for i in range(self.N):

			axker.plot(Ker[i,:],color='b')

		#axker.plot(neuron.PSP_size*mostd*neuron.synapses.ker[-1,::subsa])
		#axker.plot(neuron.PSP_size*mostd*neuron.synapses.ker[0,::subsa])
		axker.set_xlabel("time (ms)")
		axker.set_ylabel("PSP ('mV')")
		fig6.show()
		fig7 = plt.figure()
		axlls = fig7.add_subplot(111) 
		axlls.plot(self.lls,'bo')
		axlls.set_xlabel('iteration number')
		axlls.set_ylabel('log-likelihood (bits/spikes)')
		fig7.show()

		fig8 = plt.figure()
		axmd = fig8.add_subplot(gs[0,0])

		ticks = ['Poiss.','PSP (GLM)']

		for i in range(len(self.Mds)-2):

			if i%2==0:

				ticks = ticks + ['NL']

			else:

				ticks = ticks + ['PSP']

		axmd.bar(np.arange(len(self.Mds)),self.Mds,width=0.5)

		axmd.set_xticks(np.arange(len(self.Mds))+0.25)
		axmd.set_xticklabels(ticks)

		axmd.set_ylabel("Md - percentage of predicted spikes on test set.")
		
		axsw = fig8.add_subplot(gs[1,0])
		axsw.bar(np.arange(len(self.switches)),self.switches,width=0.5)

		axmd.set_xticks(np.arange(len(self.Mds)))
		axmd.set_xticklabels(ticks)

		axsw.set_ylabel("Log-likelihood (bits/spike)")
		fig8.show()

	def save(self):

		np.savetxt('paramnl.txt',self.paramNL)
		np.savetxt('paramker.txt',self.paramKer)
		np.savetxt('knots.txt',self.knots)
		np.savetxt('bnds.txt',self.bnds)


