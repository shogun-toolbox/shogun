def prepare_data(use_toy=True):
    from os.path import exists
    from tools.load import LoadMatrix
    lm=LoadMatrix()

    if not use_toy and exists('../data/../mldata/uci-20070111-optdigits.mat'):
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

    return [traindat, label_traindat, testdat, label_testdat]
