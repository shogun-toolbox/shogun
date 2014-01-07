#include <io/SerializableAsciiFile.h>
#include <lib/SGMatrix.h>
#include <features/SparseFeatures.h>
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

TEST(SparseFeaturesTest,constructor_from_dense)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), data.num_cols);

	EXPECT_EQ(features->get_sparse_feature_vector(0).num_feat_entries, 1);
	EXPECT_EQ(features->get_sparse_feature_vector(1).num_feat_entries, 2);
	EXPECT_EQ(features->get_sparse_feature_vector(2).num_feat_entries, 2);

	EXPECT_EQ(features->get_sparse_feature_vector(0).features[0].entry, data(1,0));
	EXPECT_EQ(features->get_sparse_feature_vector(1).features[0].entry, data(0,1));
	EXPECT_EQ(features->get_sparse_feature_vector(2).features[0].entry, data(0,2));

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_feature_vector_identity)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=0;
	subset_idx[1]=1;
	subset_idx[2]=2;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), data.num_cols);
	EXPECT_EQ(features->get_sparse_feature_vector(0).features[0].entry, data(1,0));
	EXPECT_EQ(features->get_sparse_feature_vector(1).features[0].entry, data(0,1));
	EXPECT_EQ(features->get_sparse_feature_vector(1).features[1].entry, data(1,1));
	EXPECT_EQ(features->get_sparse_feature_vector(2).features[0].entry, data(0,2));
	EXPECT_EQ(features->get_sparse_feature_vector(2).features[1].entry, data(1,2));

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_feature_vector_permutation)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=2;
	subset_idx[1]=0;
	subset_idx[2]=1;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), data.num_cols);
	EXPECT_EQ(features->get_sparse_feature_vector(0).features[0].entry, data(0,2));
	EXPECT_EQ(features->get_sparse_feature_vector(0).features[1].entry, data(1,2));
	EXPECT_EQ(features->get_sparse_feature_vector(1).features[0].entry, data(1,0));
	EXPECT_EQ(features->get_sparse_feature_vector(2).features[0].entry, data(0,1));
	EXPECT_EQ(features->get_sparse_feature_vector(2).features[1].entry, data(1,1));

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_feature_vector_smaller)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(2);
	subset_idx[0]=2;
	subset_idx[1]=0;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);
	EXPECT_EQ(features->get_sparse_feature_vector(0).features[0].entry, data(0,2));
	EXPECT_EQ(features->get_sparse_feature_vector(0).features[1].entry, data(1,2));
	EXPECT_EQ(features->get_sparse_feature_vector(1).features[0].entry, data(1,0));

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_full_feature_matrix_identity)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=0;
	subset_idx[1]=1;
	subset_idx[2]=2;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	SGMatrix<int32_t> mat=features->get_full_feature_matrix();
	EXPECT_TRUE(mat.equals(data));

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_full_feature_matrix_permutation)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=2;
	subset_idx[1]=0;
	subset_idx[2]=1;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	SGMatrix<int32_t> mat=features->get_full_feature_matrix();
	EXPECT_EQ(mat.num_rows, data.num_rows);
	EXPECT_EQ(mat.num_cols, subset_idx.vlen);
	for (index_t i=0; i<data.num_rows; ++i)
	{
		for (index_t j=0; j<data.num_cols; ++j)
			EXPECT_EQ(mat(i,j), data(i,subset_idx[j]));
	}

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_full_feature_matrix_repetition1)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=0;
	subset_idx[1]=1;
	subset_idx[2]=1;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	SGMatrix<int32_t> mat=features->get_full_feature_matrix();
	EXPECT_EQ(mat.num_rows, data.num_rows);
	EXPECT_EQ(mat.num_cols, subset_idx.vlen);
	for (index_t i=0; i<data.num_rows; ++i)
	{
		for (index_t j=0; j<data.num_cols; ++j)
			EXPECT_EQ(mat(i,j), data(i,subset_idx[j]));
	}

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_full_feature_matrix_smaller)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(2);
	subset_idx[0]=2;
	subset_idx[1]=1;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	SGMatrix<int32_t> mat=features->get_full_feature_matrix();
	EXPECT_EQ(mat.num_rows, data.num_rows);
	EXPECT_EQ(mat.num_cols, subset_idx.vlen);
	for (index_t i=0; i<data.num_rows; ++i)
	{
		for (index_t j=0; j<subset_idx.vlen; ++j)
			EXPECT_EQ(mat(i,j), data(i,subset_idx[j]));
	}

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_full_feature_vector_identity)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=0;
	subset_idx[1]=1;
	subset_idx[2]=2;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> vec=features->get_full_feature_vector(i);
		EXPECT_EQ(vec.vlen, data.num_rows);
		for (index_t j=0; j<vec.vlen; ++j)
			EXPECT_EQ(vec[j], data(j,subset_idx[i]));
	}

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_full_feature_vector_permutation)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=0;
	subset_idx[1]=2;
	subset_idx[2]=1;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> vec=features->get_full_feature_vector(i);
		EXPECT_EQ(vec.vlen, data.num_rows);
		for (index_t j=0; j<vec.vlen; ++j)
			EXPECT_EQ(vec[j], data(j,subset_idx[i]));
	}

	SG_UNREF(features);
}

TEST(SparseFeaturesTest,subset_get_full_feature_vector_smaller)
{
	SGMatrix<int32_t> data(2, 3);

	data(0, 0)=0;
	data(0, 1)=1;
	data(0, 2)=2;
	data(1, 0)=3;
	data(1, 1)=4;
	data(1, 2)=5;

	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	SGVector<index_t> subset_idx(2);
	subset_idx[0]=0;
	subset_idx[1]=2;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> vec=features->get_full_feature_vector(i);
		EXPECT_EQ(vec.vlen, data.num_rows);
		for (index_t j=0; j<vec.vlen; ++j)
			EXPECT_EQ(vec[j], data(j,subset_idx[i]));
	}

	SG_UNREF(features);
}
