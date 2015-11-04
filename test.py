import mechanisms
import functions

reload(mechanisms)
reload(functions)

execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()


