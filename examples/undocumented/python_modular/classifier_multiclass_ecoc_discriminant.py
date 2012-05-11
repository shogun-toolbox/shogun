from os.path import exists
from tools.load import LoadMatrix
lm=LoadMatrix()

if exists('../data/../mldata/uci-20070111-optdigits.mat'):
    from scipy.io import loadmat

    mat = loadmat('../data/../mldata/uci-20070111-optdigits.mat')['int0'].astype(float)
    X = mat[:-1,:]
    Y = mat[-1,:]
    isplit = X.shape[1]/2
    traindat = X[:,:isplit]
    label_traindat = Y[:isplit]
    testdat = X[:, isplit:]
    label_testdat = Y[isplit:]
else:
    traindat = lm.load_numbers('../data/fm_train_real.dat')
    testdat  = lm.load_numbers('../data/fm_test_real.dat')
    label_traindat = lm.load_labels('../data/label_train_multiclass.dat')
    label_testdat = None

parameter_list = [[traindat,testdat,label_traindat,label_testdat,2.1,1,1e-5],[traindat,testdat,label_traindat,label_testdat,2.2,1,1e-5]]

def classifier_multiclasslinearmachine_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,label_test_multiclass=label_testdat,lawidth=2.1,C=1,epsilon=1e-5):
    from shogun.Features import RealFeatures, Labels
    from shogun.Classifier import LibLinear, L2R_L2LOSS_SVC, LinearMulticlassMachine
    from shogun.Classifier import ECOCStrategy, ECOCDiscriminantEncoder, ECOCHDDecoder

    feats_train = RealFeatures(fm_train_real)
    feats_test  = RealFeatures(fm_test_real)

    labels = Labels(label_train_multiclass)

    classifier = LibLinear(L2R_L2LOSS_SVC)
    classifier.set_epsilon(epsilon)
    classifier.set_bias_enabled(True)

    encoder = ECOCDiscriminantEncoder()
    encoder.set_features(feats_train)
    encoder.set_labels(labels)

    strategy = ECOCStrategy(encoder, ECOCHDDecoder())

    classifier = LinearMulticlassMachine(strategy, feats_train, classifier, labels)
    classifier.train()
    label_pred = classifier.apply(feats_test)
    out = label_pred.get_labels()

    if label_test_multiclass is not None:
        from shogun.Evaluation import MulticlassAccuracy
        labels_test = Labels(label_test_multiclass)
        evaluator = MulticlassAccuracy()
        acc = evaluator.evaluate(label_pred, labels_test)
        print('Accuracy = %.4f' % acc)

    return out

if __name__=='__main__':
    print('MulticlassMachine')
    classifier_multiclasslinearmachine_modular(*parameter_list[0])

