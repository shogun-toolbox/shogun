require 'shogun'
require 'load'

traindat = load_numbers('../data/fm_train_real.dat')
testdat = load_numbers('../data/fm_test_real.dat')
label_traindat = load_labels('../data/label_train_twoclass.dat')


parameter_list = {{traindat,testdat,label_traindat,0.8,1e-6},{traindat,testdat,label_traindat,0.9,1e-7}}

function regression_krr_modular (fm_train,fm_test,label_train,width,tau)
	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)

	kernel=GaussianKernel(feats_train, feats_train, width)

	labels=Labels(label_train)

	krr=KRR(tau, kernel, labels)
	krr:train(feats_train)

	kernel:init(feats_train, feats_test)
	out = krr:apply():get_labels()
	return out,kernel,krr
end

print 'KRR'
regression_krr_modular(unpack(parameter_list[1]))
