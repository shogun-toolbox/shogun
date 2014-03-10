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
from modshogun import (LibSVMFile, GaussianKernel, MulticlassLibSVM,
		       SerializableHdf5File)
from utils import get_features_and_labels, track_execution

LOGGER = logging.getLogger(__file__)

def parse_arguments():
	parser = argparse.ArgumentParser(description="Train a multiclass SVM stored \
					 	     in libsvm format")
	parser.add_argument('--dataset', required=True, type=str,
					help='Path to training dataset in LibSVM format.')
	parser.add_argument('--capacity', default=1.0, type=float,
					help='SVM capacity parameter')
	parser.add_argument('--width', default=2.1, type=float,
					help='Width of the Gaussian Kernel to approximate')
	parser.add_argument('--epsilon', default=0.01, type=float,
					help='SVMOcas epsilon parameter')
	parser.add_argument('--output', required=True, type=str,
					help='Destination path for the output serialized \
					classifier')
	return parser.parse_args()


def main(dataset, output, epsilon, capacity, width):

	LOGGER.info("SVM Multiclass classifier")
	LOGGER.info("Epsilon: %s" % epsilon)
	LOGGER.info("Capacity: %s" % capacity)
	LOGGER.info("Gaussian width: %s" % width)

	# Get features
	feats, labels = get_features_and_labels(LibSVMFile(dataset))

	# Create kernel
	kernel = GaussianKernel(feats, feats, width)

	# Initialize and train Multiclass SVM
	svm = MulticlassLibSVM(capacity, kernel, labels)
	svm.set_epsilon(epsilon)
	with track_execution():
		svm.train()

	# Serialize to file
	writable_file = SerializableHdf5File(output, 'w')
	with closing(writable_file):
		svm.save_serializable(writable_file)
	LOGGER.info("Serialized classifier saved in: '%s'" % output)


if __name__ == '__main__':
	args = parse_arguments()
	main(args.dataset, args.output, args.epsilon, args.capacity, args.width)
