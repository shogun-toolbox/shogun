#!/usr/bin/env python
from tools.multiclass_shared import prepare_data

[traindat, label_traindat, testdat, label_testdat] = prepare_data(False)

parameter_list = [[traindat,testdat,label_traindat,label_testdat,2.1,1,1e-5],[traindat,testdat,label_traindat,label_testdat,2.2,1,1e-5]]

def classifier_multiclass_shareboost (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,label_test_multiclass=label_testdat,lawidth=2.1,C=1,epsilon=1e-5):
    from modshogun import RealFeatures, RealSubsetFeatures, MulticlassLabels
    from modshogun import ShareBoost

    #print('Working on a problem of %d features and %d samples' % fm_train_real.shape)

    feats_train = RealFeatures(fm_train_real)

    labels = MulticlassLabels(label_train_multiclass)

    shareboost = ShareBoost(feats_train, labels, min(fm_train_real.shape[0]-1, 30))
    shareboost.train();
    #print(shareboost.get_activeset())

    feats_test  = RealSubsetFeatures(RealFeatures(fm_test_real), shareboost.get_activeset())
    label_pred = shareboost.apply(feats_test)

    out = label_pred.get_labels()

    if label_test_multiclass is not None:
        from modshogun import MulticlassAccuracy
        labels_test = MulticlassLabels(label_test_multiclass)
        evaluator = MulticlassAccuracy()
        acc = evaluator.evaluate(label_pred, labels_test)
        #print('Accuracy = %.4f' % acc)

    return out

if __name__=='__main__':
    print('MulticlassMachine')
    classifier_multiclass_shareboost(*parameter_list[0])

