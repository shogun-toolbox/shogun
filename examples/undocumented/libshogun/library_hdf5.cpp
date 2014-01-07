#include <lib/config.h>
#include <base/init.h>
#include <lib/common.h>
#include <lib/SGMatrix.h>
#include <io/HDF5File.h>
#include <mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
#ifdef HAVE_HDF5
	CHDF5File* hdf = new CHDF5File((char*) "../data/australian.libsvm.h5",'r', "/data/data");
	float64_t* mat;
	int32_t num_feat;
	int32_t num_vec;
	hdf->get_matrix(mat, num_feat, num_vec);

	SGMatrix<float64_t>::display_matrix(mat, num_feat, num_vec);
	SG_FREE(mat);
	SG_UNREF(hdf);
#endif

	exit_shogun();
	return 0;
}

