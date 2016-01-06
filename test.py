execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

print "ran !"

model = TwoLayerModel()

model.add_data(neuron)

model.fit()

model.membpot() #"refresh" the membrane potential values. to compare with what the model does. because parameters have been changed by the fit but not the mp.

prob = neuron.membrane_potential

prob_mod = model.membrane_potential



h = np.histogram(prob,bins=1000.,range=[-80.,80.])
h_mod = np.histogram(prob_mod,bins=1000.,range=[-80.,80.])

fig6 = plt.figure()

ax78 = fig6.add_subplot(111)

ax78.plot(h[1][:-1],h[0])
ax78.plot(h_mod[1][:-1],h_mod[0])

V = np.arange(model.bnds[0],model.bnds[1],(model.bnds[1]-model.bnds[0])*0.00001)

Y = fun.sigmoid([-25.,50.,1.,1./25.],37.69*V)/5.

fig5 = plt.figure()

ax = fig5.add_subplot(111)

NL = np.dot(model.basisNL,model.paramNL)

ax.plot(V,Y - Y.mean() )
ax.plot(V,NL - NL.mean() )

fig5.show()





