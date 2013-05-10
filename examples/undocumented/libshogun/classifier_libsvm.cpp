/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>

using namespace shogun;

void gen_rand_data(SGVector<float64_t> lab, SGMatrix<float64_t> feat,
		float64_t dist)
{
	index_t dims=feat.num_rows;
	index_t num=lab.vlen;

	for (int32_t i=0; i<num; i++)
	{
		if (i<num/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)+dist;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)-dist;
		}
	}
	lab.display_vector("lab");
	feat.display_matrix("feat");
}

void test_libsvm()
{
	const int32_t feature_cache=0;
	const int32_t kernel_cache=0;
	const float64_t rbf_width=10;
	const float64_t svm_C=10;
	const float64_t svm_eps=0.001;

	index_t num=100;
	index_t dims=2;
	float64_t dist=0.5;

	SGVector<float64_t> lab(num);
	SGMatrix<float64_t> feat(dims, num);

	gen_rand_data(lab, feat, dist);

	// create train labels
	CLabels* labels=new CBinaryLabels(lab);

	// create train features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(
			feature_cache);
	SG_REF(features);
	features->set_feature_matrix(feat);

	// create gaussian kernel
	CGaussianKernel* kernel=new CGaussianKernel(kernel_cache, rbf_width);
	SG_REF(kernel);
	kernel->init(features, features);

	// create svm via libsvm and train
	CLibSVM* svm=new CLibSVM(svm_C, kernel, labels);
	SG_REF(svm);
	svm->set_epsilon(svm_eps);
	svm->train();

	SG_SPRINT("num_sv:%d b:%f\n", svm->get_num_support_vectors(),
			svm->get_bias());

	// classify + display output
	CBinaryLabels* out_labels=CLabelsFactory::to_binary(svm->apply());

	for (int32_t i=0; i<num; i++)
	{
		SG_SPRINT("out[%d]=%f (%f)\n", i, out_labels->get_label(i),
				out_labels->get_value(i));
	}

	SG_UNREF(out_labels);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(svm);
}

int main()
{
	init_shogun();

	test_libsvm();

	exit_shogun();
	return 0;
}

