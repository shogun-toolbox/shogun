# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindna = LoadMatrix.load_dna('../data/fm_train_dna.dat')

parameter_list = [[traindna,3],[traindna,4]]

def distribution_ppwm_modular(fm_dna=traindna, order=3)


# *** 	charfeat=StringCharFeatures(DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(DNA)
	charfeat.set_features(fm_dna)
# *** 	feats=StringByteFeatures(charfeat.get_alphabet())
	feats=Modshogun::StringByteFeatures.new
	feats.set_features(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, 0, False)

	L=20
	k=3
	sigma = 1;
	mu = 4

# *** 	ppwm=PositionalPWM()
	ppwm=Modshogun::PositionalPWM.new
	ppwm.set_features()
	ppwm.set_sigma(sigma)
	ppwm.set_mean(mu)
	pwm=array([[0.0, 0.5, 0.1, 1.0],
               [0.0, 0.5, 0.5, 0.0],
               [1.0, 0.0, 0.4, 0.0],
               [0.0, 0.0, 0.0, 0.0]]);
	pwm=array([[0.01,0.09,0.1],[0.09,0.01,0.1],[0.85,0.4,0.1],[0.05,0.5,0.7]])



	ppwm.set_pwm(log(pwm))
	#	puts ppwm.get_pwm()
	ppwm.compute_w(L)
	w=ppwm.get_w()
	#	puts w
	figure(1)
	#pcolor(exp(w))
	#pcolor(w)
	#colorbar()

	#figure(2)
	ppwm.compute_scoring(1)
	u=ppwm.get_scoring(0)
	#pcolor(exp(u))
	#show()

# *** 	#ppwm=PositionalPWM(feats)
	#ppwm=Modshogun::PositionalPWM.new
	#ppwm.set_features(feats)
	#ppwm.train()

	#out_likelihood = histo.get_log_likelihood()
	#out_sample = histo.get_log_likelihood_sample()
	return ppwm,w,u

end
###########################################################################
# call functions
###########################################################################

if __FILE__ == $0
	puts 'PositionalPWM'
	distribution_ppwm_modular(*parameter_list[0])

end
