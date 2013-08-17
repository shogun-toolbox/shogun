#include <shogun/features/DenseFeatures.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
	//sg_io->set_loglevel(MSG_DEBUG);
	//sg_io->enable_file_and_line();

	// create three 2-dimensional vectors
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++)
		matrix.matrix[i]=i;

	// shogun will now own the matrix created
	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>(matrix);

	ASSERT(features->update_parameter_hash());

	SG_UNREF(features);
	exit_shogun();

	return 0;
}
