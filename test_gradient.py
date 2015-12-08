import copy #need it to keep gradient in memory
import matplotlib.pylab as plt
import pdiff_calc as diff 

execfile('main.py') #define stuff.

neuron = TwoLayerNeuron() 
neuron.run() #generate data.

model = TwoLayerModel()
model.add_data(neuron)
state = optim.State(model) #state is kind of the "microstate" of the model. 
                           #it contains the gradients, hessians ...
Nk = np.shape(neuron.synapses.ker)[0] #number of presyn neurons
Tk = np.shape(neuron.synapses.ker)[1] #length in timesteps of the kernels.

F = state.basisKer 
Nb = np.shape(F)[0] #number of basis functions per kernel.

Ker = np.zeros((Nk,np.shape(F)[1]),dtype='float') #temporal length basis > Tk
Ker[:,:Tk] = neuron.PSP_size*neuron.synapses.ker/neuron.delta_v 

#(for above). In the model, delta_v is set to one, and accounted for by the
#absolute size of the kernels and threshold. hence the division by delta_v here.

state.paramKer[:state.N*Nb] = fun.Ker2Param(Ker,F) 

#parameters corresponding to the kernels that were defined. obtained from RSS-minim.
												   
expfun = fun.exp_fun(neuron.ASP_time,neuron.dt,neuron.ASP_total)
expfun = expfun*neuron.ASP_size
#same thing as for the PSP kernels. This way the values inside the exponential are 
# the same, even with delta_v=1

expfun = np.atleast_2d(expfun) #otherwise the Ker2Param function does not work.
basASP = state.basisASP

state.paramKer[state.N*Nb:-1] = fun.Ker2Param(expfun,basASP) #RSS minimization

state.paramNL = np.array([[-20.,40.,1.,0.05],[-20.,40.,1.,0.05]])

state.update() #update() calculates gradients and hessians from the params.

hessiank = copy.copy(state.hessian_ker)
gradientk = copy.copy(state.gradient_ker) #keep these in mem for linear approximation
paramk = copy.copy(state.paramKer)
l0 = copy.copy(state.likelihood)

res_nl = []
res_nl_l = []
res_nl_q = []

res = [] #actual likelihood curve on particular line in parameter space
res_l = [] #linear approximation with gradient
res_q = [] #quadratic approximation with gradient and hessian.

state.update()

for r in range(80,100,1):

	print r
	state.paramKer = (r/90.)*paramk #new params (line around our point)
	state.update()	# parameters have changed. Gradients and hessians are updated.
	res = res + [state.likelihood] #actual likelihood
	res_l = res_l + [l0 + np.dot(gradientk,(state.paramKer - paramk))] #linear appr

	lin_term = np.dot(gradientk,(state.paramKer - paramk))
	dif = state.paramKer - paramk
	quad_term = np.dot(dif.transpose(),np.dot(hessiank,dif))/2
	res_q = res_q + [l0 + lin_term + quad_term]

res = np.array(res)
res_l = np.array(res_l)
res_q = np.array(res_q)

plt.plot(res) 
plt.plot(res_l)
plt.plot(res_q)
plt.show()

plt.plot(res-res_l)
plt.show()

plt.plot(res-res_q)
plt.show()

