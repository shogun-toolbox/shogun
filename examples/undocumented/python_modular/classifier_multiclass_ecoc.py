import classifier_multiclass_shared
# run with toy data
[traindat, label_traindat, testdat, label_testdat] = classifier_multiclass_shared.prepare_data()
# run with opt-digits if available
#[traindat, label_traindat, testdat, label_testdat] = classifier_multiclass_shared.prepare_data(False)


import shogun.Classifier as Classifier
from shogun.Classifier import ECOCStrategy
from shogun.Features import RealFeatures, MulticlassLabels
from shogun.Classifier import LibLinear, L2R_L2LOSS_SVC, LinearMulticlassMachine
from shogun.Evaluation import MulticlassAccuracy

import re
encoders = [x for x in dir(Classifier)
        if re.match(r'ECOC.+Encoder', x)]
decoders = [x for x in dir(Classifier)
        if re.match(r'ECOC.+Decoder', x)]

fea_train = RealFeatures(traindat)
fea_test  = RealFeatures(testdat)
gnd_train = MulticlassLabels(label_traindat)
if label_testdat is None:
    gnd_test = None
else:
    gnd_test = MulticlassLabels(label_testdat)

base_classifier = LibLinear(L2R_L2LOSS_SVC)
base_classifier.set_bias_enabled(True)

print('Testing with %d encoders and %d decoders' % (len(encoders), len(decoders)))
print('-' * 70)
format_str = '%%15s + %%-10s  %%-10%s %%-10%s %%-10%s'
print((format_str % ('s', 's', 's')) % ('encoder', 'decoder', 'codelen', 'time', 'accuracy'))

def run_ecoc(ier, idr):
    encoder = getattr(Classifier, encoders[ier])()
    decoder = getattr(Classifier, decoders[idr])()

    # whether encoder is data dependent
    if hasattr(encoder, 'set_labels'):
        encoder.set_labels(gnd_train)
        encoder.set_features(fea_train)

    strategy = ECOCStrategy(encoder, decoder)
    classifier = LinearMulticlassMachine(strategy, fea_train, base_classifier, gnd_train)
    classifier.train()
    label_pred = classifier.apply(fea_test)
    if gnd_test is not None:
        evaluator = MulticlassAccuracy()
        acc = evaluator.evaluate(label_pred, gnd_test)
    else:
        acc = None

    return (classifier.get_num_machines(), acc)


import time
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
        print((format_str % ('d', '.3f', acc_fmt)) % 
                (encoders[ier][4:-7], decoders[idr][4:-7], codelen, t_elapse, acc))

