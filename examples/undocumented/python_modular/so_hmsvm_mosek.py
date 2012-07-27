#!/usr/bin/env python

import numpy
import scipy

from scipy            import io
from shogun.Features  import RealMatrixFeatures
from shogun.Loss      import HingeLoss
from shogun.Structure import HMSVMLabels, HMSVMModel, Sequence, TwoStateModel, SMT_TWO_STATE

try:
	from shogun.Structure	import PrimalMosekSOSVM
except ImportError:
	print "Mosek not available"
	import sys
	sys.exit(0)

data_dict = scipy.io.loadmat('../data/hmsvm_data_integer.mat')
labels_array = data_dict['label'][0]
idxs = numpy.nonzero(labels_array == -1)
labels_array[idxs] = 0
labels = HMSVMLabels(labels_array, 250, 100, 2)
features = RealMatrixFeatures(data_dict['signal'].astype(float), 250, 100)
loss   = HingeLoss()
model = HMSVMModel(features, labels, SMT_TWO_STATE, 4)
sosvm = PrimalMosekSOSVM(model, loss, labels, features)
sosvm.train()
print sosvm.get_w()
