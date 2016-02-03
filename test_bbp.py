execfile('main.py')

neuron = BBPneuron()

model = TwoLayerModel()

model.add_data(neuron)

model.fit()

model.membpot()

model.plot()

model.test()

model.save()

print "Md: ",model.Md
