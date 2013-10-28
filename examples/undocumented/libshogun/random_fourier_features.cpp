/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 *
 * This example demonstrates the use of the Random Fourier Dot Features with
 * a linear classifier.
 */

#include <shogun/base/init.h>
#include <shogun/features/RandomFourierDotFeatures.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/evaluation/PRCEvaluation.h>

using namespace shogun;

void load_data(int32_t num_dim, int32_t num_vecs,
	CDenseFeatures<float64_t>*& feats, CBinaryLabels*& labels)
{
	SGMatrix<float64_t> mat(num_dim, num_vecs);
	SGVector<float64_t> labs(num_vecs);
	for (index_t i=0; i<num_vecs; i++)
	{
		for (index_t j=0; j<num_dim; j++)
		{
			if ((i+j)%2==0)
			{
				labs[i] = -1;
				mat(j,i) = CMath::random(0,1) + 0.5;
			}
			else
			{
				labs[i] = 1;
				mat(j,i) = CMath::random(0,1) - 0.5;
			}
		}
	}
	feats = new CDenseFeatures<float64_t>(mat);
	labels = new CBinaryLabels(labs);
}

int main(int argv, char** argc)
{
	init_shogun_with_defaults();

	int32_t num_dim = 100;
	int32_t num_vecs = 10000;

	CDenseFeatures<float64_t>* dense_feats = 0;
	CBinaryLabels* labels = 0;
	load_data(num_dim, num_vecs, dense_feats, labels);

	/** Specifying the kernel parameter for the Gaussian approximation of RFFeatures,
	 * as specified in its documentation in KernelName.
	 * We set the kernel width of the Gaussian kernel we are approximating to 8.
	 */
	SGVector<float64_t> params(1);
	params[0] = 8;

	/** Specifying the number of samples for the RFFeatures */
	int32_t D = 300;

	/** Creating a new RandomFourierDotFeatures object, that will work on
	 * the data that we created before, will use D number of samples and
	 * will generate parameters for a Gaussian Kernel approximation of
	 * width given in params
	 */
	CRandomFourierDotFeatures* rf_feats = new CRandomFourierDotFeatures(
			dense_feats, D, KernelName::GAUSSIAN, params);

	/** Now the previous RFFeatures object can be used with a linear
	 * classifier
	 */

	//CLibLinear* lin_svm = new CLibLinear(C, r_feats, labels);
	float64_t C = 0.1;
	float64_t epsilon = 0.01;
	CSVMOcas* lin_svm = new CSVMOcas(C, rf_feats, labels);
	lin_svm->set_epsilon(epsilon);

	lin_svm->train();

	CBinaryLabels* predicted = CLabelsFactory::to_binary(lin_svm->apply());

	CPRCEvaluation* evaluator = new CPRCEvaluation();
	float64_t auPRC = evaluator->evaluate(predicted, labels);
	//SG_SPRINT("Training auPRC = %f\n", auPRC);

	SG_UNREF(lin_svm);
	SG_UNREF(predicted);
	exit_shogun();
}
