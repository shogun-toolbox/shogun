#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

int main(int argc, char** argv)
{
	//env()->io()->set_loglevel(MSG_DEBUG);
	//env()->io()->enable_file_and_line();

	// create three 2-dimensional vectors
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++)
		matrix.matrix[i]=i;

	// shogun will now own the matrix created
	DenseFeatures<float64_t>* features= new DenseFeatures<float64_t>(matrix);

	ASSERT(features->parameter_hash_changed());

	return 0;
}
