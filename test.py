execfile('main.py')

neuron = TwoLayerNeuron()

neuron.add_input()

neuron.run()

print "ran !"

model = TwoLayerModel()

model.add_data(neuron)

model.fit()

model.membpot() #"refresh" the membrane potential values. to compare with what the model does. because parameters have been changed by the fit but not the mp.

model.plot()

model.test()

print "Md: ",model.Md
	



