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
from contextlib import contextmanager, closing
from modshogun import (LibSVMFile, SparseRealFeatures, MulticlassLabels,
		       SerializableHdf5File, MulticlassAccuracy)

logging.basicConfig(level=logging.INFO, format='[%(asctime)-15s %(module)s] %(message)s')
LOGGER = logging.getLogger(__file__)

def parse_arguments():
	parser = argparse.ArgumentParser(description="Evaluate predicted labels agains \
					 	     bare truth")
	parser.add_argument('--actual', required=True, type=str,
					help='Path to LibSVM dataset.')
	parser.add_argument('--predicted', required=True, type=str,
					help='Path to serialized predicted labels')
	return parser.parse_args()

def get_features_and_labels(input_file):
	feats = SparseRealFeatures()
	label_array = feats.load_with_labels(input_file)
	labels = MulticlassLabels(label_array)
	return feats, labels


@contextmanager
def track_execution():
	LOGGER.info('Starting training.')
	timer = Time()
	yield
	timer.stop()
	LOGGER.info('Training completed, took {0:.2f}s.'.format(timer.time_diff_sec()))


def evaluate_multiclass(actual, predicted):
	LOGGER.info("SVM Multiclass evaluator")

	# Load SVMLight dataset
	feats, labels = get_features_and_labels(LibSVMFile(actual))

	# Load predicted labels
	predicted_labels = MulticlassLabels()
	predicted_labels_file = SerializableHdf5File(predicted, 'r')
	with closing(predicted_labels_file):
		predicted_labels.load_serializable(predicted_labels_file)

        multiclass_measures = MulticlassAccuracy()
        LOGGER.info("Accuracy = %s" % multiclass_measures.evaluate(
                    labels, predicted_labels))
        #LOGGER.info("Confusion matrix:")
        #res = multiclass_measures.get_confusion_matrix(labels, predicted_labels)
        #print res


def main():
	args = parse_arguments()
	evaluate_multiclass(args.actual, args.predicted)

if __name__ == '__main__':
	main()

