#include <shogun/lib/config.h>
#include <gtest/gtest.h>

#if defined(HAVE_HDF5) && defined( HAVE_CURL)
#include <shogun/io/MLDataHDF5File.h>
#include <shogun/io/HDF5File.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGMatrix.h>

#include <unistd.h>
#include <hdf5.h>


using namespace shogun;

TEST(MLDataHDF5File, read_matrix)
{
	int32_t num_rows=4;
	int32_t num_cols=3;

	float64_t data[]={1.0000, 1.0001, 1.0002, 1.0003,
					  1.0004, 1.0005, 1.0006, 1.0007,
					  1.0008, 1.0009, 1.0010, 1.0011};

	SGMatrix<float64_t> a=SGMatrix<float64_t>(data, num_rows, num_cols, false);

	char* fname = mktemp(strdup("/tmp/read_matrix.XXXXXX"));
	CHDF5File* fout=new CHDF5File(fname,'w', (char*)"/data/data");
	fout->set_matrix(data, num_rows, num_cols);
	SG_UNREF(fout);

	SGMatrix<float64_t> tmp(true);
	CMLDataHDF5File* fin;

	fin=new CMLDataHDF5File(&fname[5], "/data/data", "file:///tmp/");

	fin->get_matrix(tmp.matrix, tmp.num_rows, tmp.num_cols);
	EXPECT_EQ(tmp.num_rows, num_rows);
	EXPECT_EQ(tmp.num_cols, num_cols);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), a(i, j));
		}
	}
	SG_UNREF(fin);
	unlink(fname);
	SG_FREE(fname);
}
#else
TEST(MLDataHDF5File, DISABLED_read_matrix)
{
}
#endif
