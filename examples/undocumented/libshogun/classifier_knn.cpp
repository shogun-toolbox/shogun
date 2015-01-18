/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias García
 */

#include <shogun/base/init.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>

using namespace shogun;

#define NUM  10
#define DIMS 2
#define CLASSES 4
#define k 3

int main(int, char*[])
{
	init_shogun_with_defaults();

#ifdef HAVE_LAPACK /* because of CDataGenerator::generate_gaussians */

	// Labels and features containers
	SGVector<float64_t> lab(CLASSES*NUM);
	SGMatrix<float64_t> feat(DIMS, CLASSES*NUM);
	// Random generation of features
	feat = CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	// Labels
	for (int32_t i = 0; i < CLASSES; ++i)
		for (int32_t j = 0; j < NUM; ++j)
			lab[i*NUM + j] = i;

	// Create train labels
	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	// Create train features
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(feat);

	// Create KNN classifier
	CKNN* knn = new CKNN(k, new CEuclideanDistance(features, features), labels);
	// Train classifier
	knn->train();
	// Apply classifier
	CMulticlassLabels* output = CLabelsFactory::to_multiclass( knn->apply() );
	SGMatrix<int32_t> multiple_k_output = knn->classify_for_multiple_k();

	// Free memory
	SG_UNREF(knn)
	SG_UNREF(output)

#endif /* HAVE_LAPACK */

	exit_shogun();
	return 0;
}
