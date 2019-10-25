#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <stdio.h>

using namespace shogun;

int main(int argc, char** argv)
{
	// create some data
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++)
		matrix.matrix[i]=i;

	// create three 2-dimensional vectors
	// shogun will now own the matrix created
	DenseFeatures<float64_t>* features= new DenseFeatures<float64_t>(matrix);

	// create gaussian kernel with cache 10MB, width 0.5
	GaussianKernel* kernel = new GaussianKernel(features, features, 0.5, 10);

	// print kernel matrix
	for (int32_t i=0; i<3; i++)
	{
		for (int32_t j=0; j<3; j++)
		{
			SG_SPRINT("%f ", kernel->kernel(i,j));
		}
		SG_SPRINT("\n");
	}

	// free up memory

	return 0;
}
