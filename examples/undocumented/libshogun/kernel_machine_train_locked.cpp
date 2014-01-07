/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/init.h>
#include <features/DenseFeatures.h>
#include <labels/BinaryLabels.h>
#include <kernel/LinearKernel.h>
#include <classifier/svm/LibSVM.h>
#include <evaluation/ContingencyTableEvaluation.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void test()
{
	/* data matrix dimensions */
	index_t num_vectors=6;
	index_t num_features=2;

	/* data means -1, 1 in all components, small std deviation */
	SGVector<float64_t> mean_1(num_features);
	SGVector<float64_t> mean_2(num_features);
	SGVector<float64_t>::fill_vector(mean_1.vector, mean_1.vlen, -10.0);
	SGVector<float64_t>::fill_vector(mean_2.vector, mean_2.vlen, 10.0);
	float64_t sigma=0.5;

	SGVector<float64_t>::display_vector(mean_1.vector, mean_1.vlen, "mean 1");
	SGVector<float64_t>::display_vector(mean_2.vector, mean_2.vlen, "mean 2");

	/* fill data matrix around mean */
	SGMatrix<float64_t> train_dat(num_features, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		for (index_t j=0; j<num_features; ++j)
		{
			float64_t mean=i<num_vectors/2 ? mean_1.vector[0] : mean_2.vector[0];
			train_dat.matrix[i*num_features+j]=CMath::normal_random(mean, sigma);
		}
	}

	SGMatrix<float64_t>::display_matrix(train_dat.matrix, train_dat.num_rows, train_dat.num_cols, "training data");

	/* training features */
	CDenseFeatures<float64_t>* features=
			new CDenseFeatures<float64_t>(train_dat);
	SG_REF(features);

	/* training labels +/- 1 for each cluster */
	SGVector<float64_t> lab(num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
		lab.vector[i]=i<num_vectors/2 ? -1.0 : 1.0;

	SGVector<float64_t>::display_vector(lab.vector, lab.vlen, "training labels");

	CBinaryLabels* labels=new CBinaryLabels(lab);
	SG_REF(labels);

	/* evaluation instance */
	CContingencyTableEvaluation* eval=new CContingencyTableEvaluation(ACCURACY);

	/* kernel */
	CKernel* kernel=new CLinearKernel();
	kernel->init(features, features);

	/* create svm via libsvm */
	float64_t svm_C=10;
	float64_t svm_eps=0.0001;
	CLibSVM* svm=new CLibSVM(svm_C, kernel, labels);
	svm->set_epsilon(svm_eps);

	/* now train a few times on different subsets on data and assert that
	 * results are correct (data linear separable) */

	svm->data_lock(labels, features);

	SGVector<index_t> indices(5);
	indices.vector[0]=1;
	indices.vector[1]=2;
	indices.vector[2]=3;
	indices.vector[3]=4;
	indices.vector[4]=5;
	SGVector<index_t>::display_vector(indices.vector, indices.vlen, "training indices");
	svm->train_locked(indices);
	CBinaryLabels* output=CLabelsFactory::to_binary(svm->apply());
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "apply() output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");
	SG_SPRINT("accuracy: %f\n", eval->evaluate(output, labels));
	ASSERT(eval->evaluate(output, labels)==1);
	SG_UNREF(output);

	SG_SPRINT("\n\n");
	indices=SGVector<index_t>(3);
	indices.vector[0]=1;
	indices.vector[1]=2;
	indices.vector[2]=3;
	SGVector<index_t>::display_vector(indices.vector, indices.vlen, "training indices");
	output=CLabelsFactory::to_binary(svm->apply());
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "apply() output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");
	SG_SPRINT("accuracy: %f\n", eval->evaluate(output, labels));
	ASSERT(eval->evaluate(output, labels)==1);
	SG_UNREF(output);

	SG_SPRINT("\n\n");
	indices=SGVector<index_t>(4);
	indices.range_fill();
	SGVector<index_t>::display_vector(indices.vector, indices.vlen, "training indices");
	svm->train_locked(indices);
	output=CLabelsFactory::to_binary(svm->apply());
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "apply() output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");
	SG_SPRINT("accuracy: %f\n", eval->evaluate(output, labels));
	ASSERT(eval->evaluate(output, labels)==1);
	SG_UNREF(output);

	SG_SPRINT("normal train\n");
	svm->data_unlock();
	svm->train();
	output=CLabelsFactory::to_binary(svm->apply());
	ASSERT(eval->evaluate(output, labels)==1);
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");
	SG_UNREF(output);

	/* clean up */
	SG_UNREF(svm);
	SG_UNREF(features);
	SG_UNREF(eval);
	SG_UNREF(labels);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test();

	exit_shogun();

	return 0;
}

