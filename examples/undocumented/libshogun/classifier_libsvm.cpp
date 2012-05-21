/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

#include <stdlib.h>
#include <stdio.h>

using namespace shogun;

#define NUM 100
#define DIMS 2
#define DIST 0.5

float64_t* lab;
SGMatrix<float64_t> feat;

void gen_rand_data()
{
	lab=SG_MALLOC(float64_t, NUM);
	feat=SGMatrix<float64_t>(DIMS, NUM);

	for (int32_t i=0; i<NUM; i++)
	{
		if (i<NUM/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<DIMS; j++)
				feat.matrix[i*DIMS+j]=CMath::random(0.0,1.0)+DIST;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<DIMS; j++)
				feat.matrix[i*DIMS+j]=CMath::random(0.0,1.0)-DIST;
		}
	}
	CMath::display_vector(lab,NUM);
	CMath::display_matrix(feat.matrix,DIMS, NUM);
}

int main()
{

	const int32_t feature_cache=0;
	const int32_t kernel_cache=0;
	const float64_t rbf_width=10;
	const float64_t svm_C=10;
	const float64_t svm_eps=0.001;

	init_shogun();

	gen_rand_data();

	// create train labels
	CLabels* labels=new CBinaryLabels(SGVector<float64_t>(lab, NUM));

	// create train features
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(feature_cache);
	SG_REF(features);
	features->set_feature_matrix(feat);

	// create gaussian kernel
	CGaussianKernel* kernel = new CGaussianKernel(kernel_cache, rbf_width);
	SG_REF(kernel);
	kernel->init(features, features);

	// create svm via libsvm and train
	CLibSVM* svm = new CLibSVM(svm_C, kernel, labels);
	SG_REF(svm);
	svm->set_epsilon(svm_eps);
	svm->train();

	printf("num_sv:%d b:%f\n", svm->get_num_support_vectors(), svm->get_bias());

	// classify + display output
	CBinaryLabels* out_labels=(CBinaryLabels*) svm->apply();

	for (int32_t i=0; i<NUM; i++)
		printf("out[%d]=%f (%f)\n", i, out_labels->get_label(i), out_labels->get_confidence(i));

	SG_UNREF(out_labels);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(svm);

	exit_shogun();
	return 0;
}
