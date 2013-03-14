#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SparseFeaturesTest,serialization)
{
	/* create feature data matrix */
	SGMatrix<int32_t> data(3, 20);

	/* fill matrix with random data */
	for (index_t i=0; i<20*3; ++i)
	{
		if(i%2==0)
			data.matrix[i]=0;
		else
			data.matrix[i]=i;
	}

	//data.display_matrix();

	/* create sparse features */
	CSparseFeatures<int32_t>* sparse_features=new CSparseFeatures<int32_t>(data);

	CSerializableAsciiFile* outfile = new CSerializableAsciiFile("sparseFeatures.txt", 'w');
	sparse_features->save_serializable(outfile);
	SG_UNREF(outfile);

	CSparseFeatures<int32_t>* sparse_features_loaded = new CSparseFeatures<int32_t>();
	CSerializableAsciiFile* infile= new CSerializableAsciiFile("sparseFeatures.txt", 'r');
	sparse_features_loaded->load_serializable(infile);
	SG_UNREF(infile);

	SGMatrix<int32_t> data_loaded = sparse_features_loaded->get_full_feature_matrix();
	//data_loaded.display_matrix();

	EXPECT_TRUE(data_loaded.equals(data));

	SG_UNREF(sparse_features);
	SG_UNREF(sparse_features_loaded);
}
