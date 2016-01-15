execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

print "ran !"

model = TwoLayerModel()

model.add_data(neuron)

model.fit()

model.membpot() #"refresh" the membrane potential values. to compare with what the model does. because parameters have been changed by the fit but not the mp.

V = np.arange(model.bnds[0],model.bnds[1],(model.bnds[1]-model.bnds[0])*0.00001)

Y = fun.sigmoid([0.,25.,1.,1./25.],37.69*V)/1.

fig5 = plt.figure()

ax = fig5.add_subplot(111)


NL = np.dot(model.paramNL,model.basisNL)

ax.plot(V,Y)
ax.plot(V,NL)

fig5.show()

fig6 = plt.figure()

axker = fig6.add_subplot(111)

Ker = np.zeros((neuron.N,model.len_cos_bumps),dtype='float')

Nb = model.basisKer.shape[0]

Ker[-1,:] = np.dot(model.paramKer[:Nb],model.basisKer)

axker.plot(Ker[-1,:])
axker.plot(neuron.PSP_size*neuron.synapses.ker[-1,:model.len_cos_bumps]/38.)

fig6.show()

fig7 = plt.figure()

axlls = fig7.add_subplot(111)

axlls.plot(model.lls,'bo')

fig7.show()

model.test()

print "Md: ",model.Md
	



