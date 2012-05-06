from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat  = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_multiclasslinearmachine_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,width=2.1,C=1,epsilon=1e-5):
    from shogun.Features import RealFeatures, Labels
    from shogun.Classifier import LibLinear, L2R_L2LOSS_SVC, LinearMulticlassMachine
    from shogun.Classifier import ECOCStrategy, ECOCRandomSparseEncoder, ECOCRandomDenseEncoder, ECOCHDDecoder

    feats_train = RealFeatures(fm_train_real)
    feats_test  = RealFeatures(fm_test_real)

    labels = Labels(label_train_multiclass)

    classifier = LibLinear(L2R_L2LOSS_SVC)
    classifier.set_epsilon(epsilon)
    classifier.set_bias_enabled(True)

    rnd_dense_strategy = ECOCStrategy(ECOCRandomDenseEncoder(), ECOCHDDecoder())
    rnd_sparse_strategy = ECOCStrategy(ECOCRandomSparseEncoder(), ECOCHDDecoder())

    dense_classifier = LinearMulticlassMachine(rnd_dense_strategy, feats_train, classifier, labels)
    dense_classifier.train()
    out_dense = dense_classifier.apply(feats_test).get_labels()

    sparse_classifier = LinearMulticlassMachine(rnd_sparse_strategy, feats_train, classifier, labels)
    sparse_classifier.train()
    out_sparse = sparse_classifier.apply(feats_test).get_labels()

    return out_sparse, out_dense

if __name__=='__main__':
    print('MulticlassMachine')
    classifier_multiclasslinearmachine_modular(*parameter_list[0])

