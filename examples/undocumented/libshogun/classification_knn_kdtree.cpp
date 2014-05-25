/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011 Sergey Lisitsyn
 * Written (W) 2012 Fernando José Iglesias García, cover tree support
 * Written (W) 2014 Zharmagambetov Arman, kd tree support
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */


#include <iostream>
#include <vector>

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/multiclass/KNN.h>

using namespace shogun;

void gen_rand_data(SGVector<float64_t> &lab, SGMatrix<float64_t> &feat) {
	index_t dims = feat.num_rows;
	index_t num = lab.vlen;

	for (int32_t i = 0; i < num; i++) {
		if (i < num / 2) {
			lab[i] = 0;

			for (int32_t j = 0; j < dims; j++)
				feat(j, i) = CMath::random(-10.0, 10.0);
		} else {
			lab[i] = 1.0;

			for (int32_t j = 0; j < dims; j++)
				feat(j, i) = CMath::random(-10.0, 10.0);
		}
	}
}

int main() {
	init_shogun_with_defaults();

	index_t num = 6;
	index_t dim = 2;

	SGMatrix<float64_t> features(dim, num);

	SGVector<float64_t> labels(num);

	gen_rand_data(labels, features);

	CMulticlassLabels* multilabels = new CMulticlassLabels(labels);

	CDenseFeatures<float64_t>* denseFeatures = new CDenseFeatures<float64_t>(
			features);

	CKNN* kdtree = new CKNN(3,
			new CEuclideanDistance(denseFeatures, denseFeatures), multilabels);
	kdtree->set_mode(kdtree->KDTree);
	kdtree->train();

	SGMatrix<float64_t> test(2, 1);
	test(0, 0) = 4.0;
	test(1, 0) = 3.0;

	CDenseFeatures<float64_t>* testFeatures = new CDenseFeatures<float64_t>(
			test);

	CMulticlassLabels* multiTestLab = kdtree->apply_multiclass(testFeatures);
	SGVector<int32_t> lab =
			((CMulticlassLabels*) multiTestLab)->get_int_labels();

	SG_SPRINT("The classification: %d \n", lab[0]);

	SG_UNREF(kdtree);
	exit_shogun();

	return 0;
}
