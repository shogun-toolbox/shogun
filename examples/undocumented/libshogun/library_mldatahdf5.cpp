#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/io/MLDataHDF5File.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
#if defined(HAVE_HDF5) && defined( HAVE_CURL)
	MLDataHDF5File* hdf = NULL;
	try
	{
		hdf = new MLDataHDF5File((char *)"australian", "/data/data");
	}
	catch (ShogunException& e)
	{
		return 0;
	}
	float64_t* mat=NULL;
	int32_t num_feat;
	int32_t num_vec;
	try
	{
		hdf->get_matrix(mat, num_feat, num_vec);
		SGMatrix<float64_t>::display_matrix(mat, num_feat, num_vec);
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s", e.what());
	}

	SG_FREE(mat);
#endif // HAVE_CURL && HAVE_HDF5
	return 0;
}
