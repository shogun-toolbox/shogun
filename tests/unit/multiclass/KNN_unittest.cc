/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias García
 * Written (W) 2014 Dhruv Jawali
 */

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <gtest/gtest.h>

using namespace shogun;

#define NUM  10
#define DIMS 2
#define CLASSES 4
#define k 3

#ifdef HAVE_LAPACK 
TEST(KNN, BruteForce)
{
	SGVector<float64_t> lab(CLASSES*NUM);
	SGMatrix<float64_t> feat(DIMS, CLASSES*NUM);
	SGMatrix<float64_t> dat(DIMS, 10);

	feat = CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	feat = CDataGenerator::generate_gaussians(10,CLASSES,DIMS);

	for (int32_t i = 0; i < CLASSES; ++i)
		for (int32_t j = 0; j < NUM; ++j)
			lab[i*NUM + j] = i;

	CMulticlassLabels* labels = new CMulticlassLabels(lab);

	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(feat);
	CDenseFeatures<float64_t>* data = new CDenseFeatures<float64_t>(dat);
	
	CKNN* knn = new CKNN(k, new CEuclideanDistance(features, features), labels);
	knn->train();
	CMulticlassLabels* output = CLabelsFactory::to_multiclass( knn->apply(data) );
	
	SG_UNREF(knn)
	SG_UNREF(output)
}

TEST(KNN, CoverTree)
{
	SGVector<float64_t> lab(CLASSES*NUM);
	SGMatrix<float64_t> feat(DIMS, CLASSES*NUM);
	SGMatrix<float64_t> dat(DIMS, 10);

	feat = CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	feat = CDataGenerator::generate_gaussians(10,CLASSES,DIMS);

	for (int32_t i = 0; i < CLASSES; ++i)
		for (int32_t j = 0; j < NUM; ++j)
			lab[i*NUM + j] = i;

	CMulticlassLabels* labels = new CMulticlassLabels(lab);

	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(feat);
	CDenseFeatures<float64_t>* data = new CDenseFeatures<float64_t>(dat);
	
	CKNN* knn = new CKNN(k, new CEuclideanDistance(features, features), labels, CKNN::CoverTree);
	knn->train();
	CMulticlassLabels* output = CLabelsFactory::to_multiclass( knn->apply(data) );
	
	SG_UNREF(knn)
	SG_UNREF(output)
}

TEST(KNN, KDTree)
{
	SGVector<float64_t> lab(CLASSES*NUM);
	SGMatrix<float64_t> feat(DIMS, CLASSES*NUM);
	SGMatrix<float64_t> dat(DIMS, 10);

	feat = CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	feat = CDataGenerator::generate_gaussians(10,CLASSES,DIMS);

	for (int32_t i = 0; i < CLASSES; ++i)
		for (int32_t j = 0; j < NUM; ++j)
			lab[i*NUM + j] = i;

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(feat);
	CDenseFeatures<float64_t>* data = new CDenseFeatures<float64_t>(dat);
	
	CKNN* knn = new CKNN(k, new CEuclideanDistance(features, features), labels, CKNN::KDTree, 20);
	knn->train();
	CMulticlassLabels* output = CLabelsFactory::to_multiclass( knn->apply(data) );
	
	SG_UNREF(knn)
	SG_UNREF(output)
}
#endif
