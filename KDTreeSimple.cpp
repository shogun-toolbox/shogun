//============================================================================
// Name        : KDTreeSimple.cpp
// Author      : Arman Zharmagambetov armanform@gmail.com
// Version     :
// Copyright   : This program is free software and you can redistribute it and/or modify
//				 it under the terms of the GNU General Public License
// Description : This class implements simple KD tree structure without integration with KNN.
//============================================================================

#include <iostream>
#include <vector>
#include "KDTree.h"
#include <shogun/base/init.h>

using namespace std;
using namespace shogun;


int main()
{
	init_shogun_with_defaults();
	KDTree<float64_t> kdtree;

	SGMatrix<float64_t> features(2,6);
	features(0,0) = 1;
	features(1,0) = 2;

	features(0,1) = 6;
	features(1,1) = 3;

	features(0,2) = 2;
	features(1,2) = 1;

	features(0,3) = 4;
	features(1,3) = 1;

	features(0,4) = 3;
	features(1,4) = 3;

	features(0,5) = 1;
	features(1,5) = 4;

	SGVector<float64_t> labels;
	labels[0] = 0;
	labels[1] = 0;
	labels[2] = 1;
	labels[3] = 0;
	labels[4] = 1;
	labels[5] = 1;

	//construct
	kdtree.train_kd(features, labels);


	vector<float64_t> nntest;
	nntest.push_back(5);
	nntest.push_back(2);

	// in order to test classify test vector 
	// we apply for k nierest neighbor
	int get_class = kdtree.apply_one(nntest, 3);

	cout << get_class << endl;

	exit_shogun();
	return 0;
}
