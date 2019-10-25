/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Roman Votyakov, 
 *          Sergey Lisitsyn
 */

#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

/* number of features and their dimension, number of kernels */
int main(int argc, char** argv)
{
	const int32_t n=7;

	/* create some random data and hand it to each kernel */
	SGMatrix<float64_t> matrix(n,n);
	for (int32_t k=0; k<n*n; ++k)
		matrix.matrix[k]=Math::random((float64_t) -n, (float64_t) n);

	SG_SPRINT("feature data:\n");
	SGMatrix<float64_t>::display_matrix(matrix.matrix, n, n);

	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(matrix);

	/* create n kernels with n features each */
	GaussianKernel** kernels=SG_MALLOC(GaussianKernel*, n);
	for (int32_t i=0; i<n; ++i)
	{
		kernels[i]=new GaussianKernel(10, Math::random(0.0, (float64_t)n*n));

		/* hand data to kernel */
		kernels[i]->init(features, features);
	}

	/* create n parameter instances, each with one kernel */
	Parameter** parameters=SG_MALLOC(Parameter*, n);
	for (int32_t i=0; i<n; ++i)
	{
		parameters[i]=new Parameter();
		parameters[i]->add((SGObject**)&kernels[i], "kernel", "");
	}

	/* create n labels (+1,-1,+1,-1,...) */
	BinaryLabels* labels=new BinaryLabels(n);
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
	return 0;
}
