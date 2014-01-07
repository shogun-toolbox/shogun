/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <labels/BinaryLabels.h>
#include <features/DenseFeatures.h>
#include <kernel/GaussianKernel.h>
#include <classifier/svm/LibSVM.h>
#include <base/init.h>
#include <lib/common.h>
#include <io/SGIO.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

/* number of features and their dimension, number of kernels */
int main(int argc, char** argv)
{
	const int32_t n=7;

	init_shogun(&print_message);

	/* create some random data and hand it to each kernel */
	SGMatrix<float64_t> matrix(n,n);
	for (int32_t k=0; k<n*n; ++k)
		matrix.matrix[k]=CMath::random((float64_t) -n, (float64_t) n);

	SG_SPRINT("feature data:\n");
	SGMatrix<float64_t>::display_matrix(matrix.matrix, n, n);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(matrix);

	/* create n kernels with n features each */
	CGaussianKernel** kernels=SG_MALLOC(CGaussianKernel*, n);
	for (int32_t i=0; i<n; ++i)
	{
		kernels[i]=new CGaussianKernel(10, CMath::random(0.0, (float64_t)n*n));

		/* hand data to kernel */
		kernels[i]->init(features, features);
	}

	/* create n parameter instances, each with one kernel */
	Parameter** parameters=SG_MALLOC(Parameter*, n);
	for (int32_t i=0; i<n; ++i)
	{
		parameters[i]=new Parameter();
		parameters[i]->add((CSGObject**)&kernels[i], "kernel", "");
	}

	/* create n labels (+1,-1,+1,-1,...) */
	CBinaryLabels* labels=new CBinaryLabels(n);
	for (int32_t i=0; i<n; ++i)
		labels->set_label(i, i%2==0 ? +1 : -1);

	/* create libsvm with C=10 and produced labels */
	CLibSVM* svm=new CLibSVM(10, NULL, labels);

	/* iterate over all parameter instances and set them as subkernel */
	for (int32_t k=0; k<n; ++k)
	{
		SG_SPRINT("\nkernel %d has width %f\n", k, kernels[k]->get_width());

		/* change kernel, old one is UNREF'ed, new one is REF'ed */
		svm->m_parameters->set_from_parameters(parameters[k]);

		/* train and classify with the different kernels */
		svm->train();
		for (int32_t i=0; i<n; ++i)
			SG_SPRINT("output[%d]=%f\treal[%d]=%f\n", i,
					svm->apply_one(i), i, labels->get_label(i));
	}

	/* free up memory: delete all Parameter instances */
	for (int32_t i=0; i<n; ++i)
		delete parameters[i];

	/* delete created arrays */
	SG_FREE(kernels);
	SG_FREE(parameters);

	/* this also handles features, labels, and last kernel in kernels[n-1] */
	SG_UNREF(svm);

	exit_shogun();
	return 0;
}
