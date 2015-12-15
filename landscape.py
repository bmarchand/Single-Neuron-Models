execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

print "ran !"

model = TwoLayerModel()

model.add_data(neuron)

grid_c = np.arange(0.2,2.8,0.05)

LLs = np.zeros((grid_c.size),dtype='float')

state = optim.State(model)

state.paramKer = np.loadtxt('paramKer_ref.txt')

for c in range(grid_c.size):

	state.paramNL[0][2] = grid_c[c]

	state.update()

	LLs[c] = np.sign(state.likelihood)*math.log(abs(state.likelihood))

plt.plot(LLs)
plt.show()
