#!/usr/bin/env python

#  Copyright (c) The Shogun Machine Learning Toolbox
#  Written (w) 2014 Daniel Pyrathon
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
#  The views and conclusions contained in the software and documentation are those
#  of the authors and should not be interpreted as representing official policies,
#  either expressed or implied, of the Shogun Development Team.


import argparse
import logging
import numpy as np
from modshogun import (LibSVMFile, MulticlassLabels, MulticlassAccuracy)
from utils import get_features_and_labels

LOGGER = logging.getLogger(__file__)

def parse_arguments():
	parser = argparse.ArgumentParser(description="Evaluate predicted \
					labels againsy bare truth")
	parser.add_argument('--actual', required=True, type=str,
					help='Path to LibSVM dataset.')
	parser.add_argument('--predicted', required=True, type=str,
					help='Path to serialized predicted labels')
	return parser.parse_args()


def main(actual, predicted):
	LOGGER.info("SVM Multiclass evaluator")

	# Load SVMLight dataset
	feats, labels = get_features_and_labels(LibSVMFile(actual))

	# Load predicted labels
	with open(predicted, 'r') as f:
		predicted_labels_arr = np.array([float(l) for l in f])
		predicted_labels = MulticlassLabels(predicted_labels_arr)

	# Evaluate accuracy
	multiclass_measures = MulticlassAccuracy()
	LOGGER.info("Accuracy = %s" % multiclass_measures.evaluate(
		labels, predicted_labels))
	LOGGER.info("Confusion matrix:")
	res = multiclass_measures.get_confusion_matrix(labels, predicted_labels)
	print res


if __name__ == '__main__':
	args = parse_arguments()
	main(args.actual, args.predicted)
