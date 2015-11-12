import mechanisms
import functions
import diff

t0 = time.time()

reload(mechanisms)
reload(functions)

execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

print "output: ",neuron.output_rate," Hz"
print "run time: ",time.time()-t0, " sec."


