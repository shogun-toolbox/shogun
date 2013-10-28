#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna,3],[traindna,4]]

def distribution_ppwm_modular (fm_dna=traindna, order=3):
	from modshogun import StringByteFeatures, StringCharFeatures, DNA
	from modshogun import PositionalPWM

	from numpy import array,e,log,exp

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_dna)
	feats=StringByteFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, 0, False)

	L=20
	k=3
	sigma = 1;
	mu = 4

	ppwm=PositionalPWM()
	ppwm.set_sigma(sigma)
	ppwm.set_mean(mu)
	pwm=array([[0.0, 0.5, 0.1, 1.0],
               [0.0, 0.5, 0.5, 0.0],
               [1.0, 0.0, 0.4, 0.0],
               [0.0, 0.0, 0.0, 0.0]]);
	pwm=array([[0.01,0.09,0.1],[0.09,0.01,0.1],[0.85,0.4,0.1],[0.05,0.5,0.7]])



	ppwm.set_pwm(log(pwm))
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
	distribution_ppwm_modular(*parameter_list[0])
