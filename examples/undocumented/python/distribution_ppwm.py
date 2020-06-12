#!/usr/bin/env python
import shogun as sg
import numpy as np
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna,3],[traindna,4]]

def distribution_ppwm (fm_dna=traindna, order=3):

	charfeat=sg.create_string_features(fm_dna, sg.DNA)
	feats=sg.create_string_features(charfeat, order-1, order, 0, False, sg.PT_UINT8)
	feats.put("alphabet", sg.as_alphabet(charfeat.get("alphabet")))

	L=20
	k=3
	sigma = 1;
	mu = 4

	ppwm=sg.PositionalPWM()
	ppwm.set_sigma(sigma)
	ppwm.set_mean(mu)
	pwm=np.array([[0.0, 0.5, 0.1, 1.0],
               [0.0, 0.5, 0.5, 0.0],
               [1.0, 0.0, 0.4, 0.0],
               [0.0, 0.0, 0.0, 0.0]]);
	pwm=np.array([[0.01,0.09,0.1],[0.09,0.01,0.1],[0.85,0.4,0.1],[0.05,0.5,0.7]])

	ppwm.set_pwm(np.log(pwm))
	#print(ppwm.get_pwm())
	ppwm.compute_w(L)
	w=ppwm.get_w()
	#print(w)
	#from pylab import *
	#figure(1)
	#pcolor(exp(w))
	#pcolor(w)
	#colorbar()

	#figure(2)
	ppwm.compute_scoring(1)
	u=ppwm.get_scoring(0)
	#pcolor(exp(u))
	#show()

	#ppwm=PositionalPWM(feats)
	#ppwm.train()

	#out_likelihood = histo.get_log_likelihood()
	#out_sample = histo.get_log_likelihood_sample()
	return w,u
###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	print('PositionalPWM')
	distribution_ppwm(*parameter_list[0])
