/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann and others
 */

#include <labels/MulticlassLabels.h>
#include <features/DenseFeatures.h>
#include <kernel/GaussianKernel.h>
#include <multiclass/LaRank.h>
#include <base/init.h>

using namespace shogun;

void test()
{
	index_t num_vec=10;
	index_t num_feat=3;
	index_t num_class=num_feat; // to make data easy
	float64_t distance=15;

	// create some linearly seperable data
	SGMatrix<float64_t> matrix(num_class, num_vec);
	SGMatrix<float64_t> matrix_test(num_class, num_vec);
	CMulticlassLabels* labels=new CMulticlassLabels(num_vec);
	CMulticlassLabels* labels_test=new CMulticlassLabels(num_vec);
	for (index_t i=0; i<num_vec; ++i)
	{
		index_t label=i%num_class;
		for (index_t j=0; j<num_feat; ++j)
		{
			matrix(j,i)=CMath::randn_double();
			matrix_test(j,i)=CMath::randn_double();
			labels->set_label(i, label);
			labels_test->set_label(i, label);
		}

		/* make sure data is linearly seperable per class */
		matrix(label,i)+=distance;
		matrix_test(label,i)+=distance;
	}
	matrix.display_matrix("matrix");
	labels->get_int_labels().display_vector("labels");

	// shogun will now own the matrix created
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(matrix);
	CDenseFeatures<float64_t>* features_test=
			new CDenseFeatures<float64_t>(matrix_test);

	// create three labels
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	// create gaussian kernel with cache 10MB, width 0.5
	CGaussianKernel* kernel = new CGaussianKernel(10, 0.5);
	kernel->init(features, features);

	// create libsvm with C=10 and train
	CLaRank* svm = new CLaRank(10, kernel, labels);
	svm->train();
	svm->train();

	// classify on training examples
	CMulticlassLabels* output=(CMulticlassLabels*)svm->apply();
	output->get_labels().display_vector("batch output");

	/* assert that batch apply and apply(index_t) give same result */
	SGVector<float64_t> single_outputs(output->get_num_labels());
	for (index_t i=0; i<output->get_num_labels(); ++i)
		single_outputs[i]=svm->apply_one(i);

	single_outputs.display_vector("single_outputs");

	for (index_t i=0; i<output->get_num_labels(); ++i)
		ASSERT(output->get_label(i)==single_outputs[i]);

	CMulticlassLabels* output_test=
			(CMulticlassLabels*)svm->apply(features_test);
	labels_test->get_labels().display_vector("labels_test");
	output_test->get_labels().display_vector("output_test");

	for (index_t i=0; i<output->get_num_labels(); ++i)
		ASSERT(labels_test->get_label(i)==output_test->get_label(i));

	// free up memory
	SG_UNREF(output);
	SG_UNREF(labels_test);
	SG_UNREF(output_test);
	SG_UNREF(svm);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();
	return 0;
}

