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
from contextlib import closing
from modshogun import (LibSVMFile, SparseRealFeatures, MulticlassLabels,
											MulticlassLibSVM, SerializableHdf5File,
											MulticlassAccuracy)
from utils import get_features_and_labels

LOGGER = logging.getLogger(__file__)

def parse_arguments():
	parser = argparse.ArgumentParser(description="Test a serialized SVM \
					classifier agains a SVMLight test file")
	parser.add_argument('--classifier', required=True, type=str,
					help='Path to training dataset in LibSVM format.')
	parser.add_argument('--testset', required=True, type=str,
					help='Path to the SVMLight test file')
	parser.add_argument('--output', required=True, type=str,
					help='File path to write predicted labels')
	return parser.parse_args()


def main(classifier, testset, output):
	LOGGER.info("SVM Multiclass evaluation")

	svm = MulticlassLibSVM()
	serialized_classifier = SerializableHdf5File(classifier, 'r')
	with closing(serialized_classifier):
		svm.load_serializable(serialized_classifier)

	test_feats, test_labels = get_features_and_labels(LibSVMFile(testset))
	predicted_labels = svm.apply(test_feats)

	with open(output, 'w') as f:
		for cls in predicted_labels.get_labels():
			f.write("%s\n" % int(cls))

	LOGGER.info("Predicted labels saved in: '%s'" % output)


if __name__ == '__main__':
	args = parse_arguments()
	main(args.classifier, args.testset, args.output)

