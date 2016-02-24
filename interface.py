import os
import numpy as np

def import_data(l_indices,path,strs,grps):

	os.chdir(path)

	output = np.loadtxt('spikes_'+str(l_indices[0])+'.txt')
	
	output_test = np.loadtxt('spikes_'+str(l_indices[1])+'.txt')

	inputs = []

	input_test = []

	for i in range(len(grps)):

		for j in grps[i]:

			name = 'vecs_spks_'+str(l_indices[0])+'_'+str(j)+'.txt'

			name_test = 'vecs_spks_'+str(l_indices[1])+'_'+str(j)+'.txt'

			inputs = inputs + [list(np.loadtxt(name))]

			input_test = input_test + [list(np.loadtxt(name_test))]

	os.chdir('/home/bmarchan/LNLN/')

	return input_test, inputs, output_test, output

