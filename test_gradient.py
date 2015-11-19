execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

model = TwoLayerModel()

model.add_data(neuron)

state = optim.State(model)

Ker = neuron.synapses.ker

F = state.basisKer

Nb = np.shape(F)[0]

state.paramKer[:state.N*Nb] = fun.Ker2Param(Ker,F)

state.paramKer[state.N*Nb:] = fun.Ker2Param()

state.update()





