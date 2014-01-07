#include <lib/config.h>
#include <base/init.h>
#include <lib/common.h>
#include <lib/SGMatrix.h>
#include <io/MLDataHDF5File.h>
#include <io/SGIO.h>
#include <mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
#if defined(HAVE_HDF5) && defined( HAVE_CURL)
	CMLDataHDF5File* hdf = new CMLDataHDF5File((char *)"australian", "/data/data");
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
		SG_SWARNING("%s", e.get_exception_string());
	}

	SG_FREE(mat);
	SG_UNREF(hdf);
#endif // HAVE_CURL && HAVE_HDF5
	exit_shogun();
	return 0;
}
