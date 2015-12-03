execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

print "ran !"

model = TwoLayerModel()

model.add_data(neuron)

model.fit()

model.plot()
