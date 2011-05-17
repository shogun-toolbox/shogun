/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/Labels.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/io.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

/* number of features and their dimension */
int32_t n=6;

int main(int argc, char** argv)
{
	init_shogun(&print_message);

	/* create some random data */
	float64_t* matrix = new float64_t[n*n];
	for(int32_t i=0; i<n*n; ++i)
		matrix[i]=CMath::random((float64_t)-n,(float64_t)n);

	CMath::display_matrix(matrix, n, n);

	/* create n n-dimensional feature vectors */
	CSimpleFeatures<float64_t>* features= new CSimpleFeatures<float64_t>();
	features->set_feature_matrix(matrix, n, n);

	/* create gaussian kernel with cache 10MB, width will be changed later */
	CGaussianKernel* kernel = new CGaussianKernel(10, 0);
	kernel->init(features, features);

	/* create n labels (+1,-1,+1,-1,...) */
	CLabels* labels=new CLabels(n);
	for (int32_t i=0; i<n; ++i)
		labels->set_label(i, i%2==0 ? +1 : -1);

	/* create libsvm with C=10 and produced labels */
	CLibSVM* svm=new CLibSVM(10, kernel, labels);

	/* iterate over different width parameters */
	for (int32_t i=0; i<10; ++i)
	{
		SG_SPRINT("\n\ncurrent kernel width: 2^%d=%f\n", i, CMath::pow(2.0,i));

		float64_t width=CMath::pow(2.0,i);

		/* create parameter to change current kernel width */
		Parameter* param=new Parameter();
		param->add(&width, "width", "");

		/* tell kernel to use the newly produced parameter */
		kernel->m_parameters->set_from_parameters(param);

		/* print kernel matrix */
		for (int32_t i=0; i<n; i++)
		{
			for (int32_t j=0; j<n; j++)
				SG_SPRINT("%f ", kernel->kernel(i,j));

			SG_SPRINT("\n");
		}

		/* train and classify */
		svm->train();
		for (int32_t i=0; i<n; ++i)
			SG_SPRINT("output[%d]=%f\treal[%d]=%f\n", i,
					svm->classify_example(i), i, labels->get_label(i));

		delete param;
	}

	/* free up memory */
	SG_UNREF(svm);

	exit_shogun();
	return 0;
}
