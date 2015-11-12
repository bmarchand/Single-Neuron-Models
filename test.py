import mechanisms
import functions
import diff

t0 = time.time()

reload(mechanisms)
reload(functions)

execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

model = TwoLayerModel()

model.add_data(neuron)

model.fit()

model.plot()