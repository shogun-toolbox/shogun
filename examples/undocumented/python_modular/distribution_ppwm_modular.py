from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna,3],[traindna,4]]

def distribution_ppwm_modular (fm_dna=traindna, order=3):
	from shogun.Features import StringByteFeatures, StringCharFeatures, DNA
	from shogun.Distribution import PositionalPWM

	from numpy import array,e,log,exp

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_dna)
	feats=StringByteFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, 0, False)

	ppwm=PositionalPWM()
	ppwm.set_sigma(5.0)
	ppwm.set_mean(10.0)
	pwm=array([[0.0, 0.5, 0.1, 1.0],
               [0.0, 0.5, 0.5, 0.0],
               [1.0, 0.0, 0.4, 0.0],
               [0.0, 0.0, 0.0, 0.0]]);
	ppwm.set_pwm(log(pwm))
	#print ppwm.get_pwm()
	ppwm.compute_w(20)
	w= ppwm.get_w()
	#print w
	#from pylab import *
	#pcolor(exp(w))
	#show()

	#ppwm=PositionalPWM(feats)
	#ppwm.train()

	#out_likelihood = histo.get_log_likelihood()
	#out_sample = histo.get_log_likelihood_sample()
	#return histo,out_sample,out_likelihood
###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	print 'PositionalPWM'
	distribution_ppwm_modular(*parameter_list[0])

