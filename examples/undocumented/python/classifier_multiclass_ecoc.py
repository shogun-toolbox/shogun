#!/usr/bin/env python
import re
import time
from tools.multiclass_shared import prepare_data

# run with toy data
[traindat, label_traindat, testdat, label_testdat] = prepare_data()
# run with opt-digits if available
#[traindat, label_traindat, testdat, label_testdat] = prepare_data(False)

parameter_list = [[traindat,testdat,label_traindat,label_testdat,2.1,1,1e-5]]

def classifier_multiclass_ecoc (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,label_test_multiclass=label_testdat,lawidth=2.1,C=1,epsilon=1e-5):

	import shogun as sg

	def nonabstract_class(name):
		try:
		    getattr(sg, name)()
		except TypeError:
		    return False
		return True

	encoders = [x for x in dir(sg)
		    if re.match(r'ECOC.+Encoder', x) and nonabstract_class(x)]
	decoders = [x for x in dir(sg)
		    if re.match(r'ECOC.+Decoder', x) and nonabstract_class(x)]

	fea_train = sg.create_features(fm_train_real)
	fea_test  = sg.create_features(fm_test_real)
	gnd_train = sg.create_labels(label_train_multiclass)
	if label_test_multiclass is None:
		gnd_test = None
	else:
		gnd_test = sg.create_labels(label_test_multiclass)

	base_classifier = sg.create_machine("LibLinear",
								liblinear_solver_type="L2R_L2LOSS_SVC",
								use_bias=True)

	#print('Testing with %d encoders and %d decoders' % (len(encoders), len(decoders)))
	#print('-' * 70)
	#format_str = '%%15s + %%-10s  %%-10%s %%-10%s %%-10%s'
	#print((format_str % ('s', 's', 's')) % ('encoder', 'decoder', 'codelen', 'time', 'accuracy'))

	def run_ecoc(ier, idr):
		encoder = sg.create_ecoc_encoder(encoders[ier])
		decoder = sg.create_ecoc_decoder(decoders[idr])

		# whether encoder is data dependent
		if encoder.has('labels'):
		    encoder.put('labels', gnd_train)
		    encoder.put('features', fea_train)

		strategy = sg.ECOCStrategy(encoder, decoder)
		classifier = sg.create_machine("LinearMulticlassMachine", multiclass_strategy=strategy, 
										machine=base_classifier, labels=gnd_train)
		classifier.train(fea_train)
		label_pred = classifier.apply(fea_test)
		if gnd_test is not None:
		    evaluator = sg.create_evaluation("MulticlassAccuracy")
		    acc = evaluator.evaluate(label_pred, gnd_test)
		else:
		    acc = None

		return (len(classifier.get("machines")), acc)


	for ier in range(len(encoders)):
		for idr in range(len(decoders)):
		    t_begin = time.clock()
		    (codelen, acc) = run_ecoc(ier, idr)
		    if acc is None:
		        acc_fmt = 's'
		        acc = 'N/A'
		    else:
		        acc_fmt = '.4f'

		    t_elapse = time.clock() - t_begin
		    #print((format_str % ('d', '.3f', acc_fmt)) %
		    #        (encoders[ier][4:-7], decoders[idr][4:-7], codelen, t_elapse, acc))

if __name__=='__main__':
    print('MulticlassECOC')
    classifier_multiclass_ecoc(*parameter_list[0])

