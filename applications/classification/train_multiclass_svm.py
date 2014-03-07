#!/usr/bin/env python

#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  Written (W) 2014 Daniel Pyrathon
#

import argparse
from modshogun import LibSVMFile, SparseRealFeatures, MulticlassLabels, GaussianKernel, MulticlassLibSVM, SerializableHdf5File

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
	input_file = LibSVMFile(dataset)

	sparse_feats = SparseRealFeatures()
	label_array = sparse_feats.load_with_labels(input_file)
	labels = MulticlassLabels(label_array)

	kernel = GaussianKernel(sparse_feats , sparse_feats, width)

	svm = MulticlassLibSVM(capacity, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.train()

	writable_file = SerializableHdf5File(output, 'w')
	svm.save_serializable(writable_file)
	print "DONE"


if __name__ == '__main__':
	args = parse_arguments()
	main(args.dataset, args.output, args.epsilon, args.capacity, args.width)
