execfile('main.py')

neuron = TwoLayerNeuron()

neuron.run()

print "ran !"

model = TwoLayerModel()

model.add_data(neuron)

model.fit()

model.membpot()

prob = neuron.membrane_potential

prob_mod = model.membrane_potential

fig5 = plt.figure()

ax = fig5.add_subplot(111)

ax.plot(prob)
ax.plot(prob_mod)

fig5.show()

h = np.histogram(prob)
h_mod = np.histogram(prob_mod)

fig6 = plt.figure()

ax78 = fig6.add_subplot(111)

ax78.plot(h[1][:-1],h[0])
ax78.plot(h_mod[1][:-1],h_mod[0])






