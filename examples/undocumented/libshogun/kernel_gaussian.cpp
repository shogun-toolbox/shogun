#include <shogun/features/SimpleFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <stdio.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message);

	// create some data
	float64_t* matrix = SG_MALLOCX(float64_t, 6);
	for (int32_t i=0; i<6; i++)
		matrix[i]=i;

	// create three 2-dimensional vectors 
	// shogun will now own the matrix created
	CSimpleFeatures<float64_t>* features= new CSimpleFeatures<float64_t>();
	features->set_feature_matrix(matrix, 2, 3);

	// create gaussian kernel with cache 10MB, width 0.5
	CGaussianKernel* kernel = new CGaussianKernel(features, features, 10, 0.5);

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
	SG_UNREF(kernel);

	exit_shogun();
	return 0;
}
