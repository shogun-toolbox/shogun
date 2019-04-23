/*
* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <string>

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
	auto sparse_features=std::make_shared<SparseFeatures<int32_t>>(data);

	auto fs = env();
	std::string filename("sparseFeatures.json");
	ASSERT_TRUE(fs->file_exists(filename));
	std::unique_ptr<io::WritableFile> file;
	ASSERT_FALSE(fs->new_writable_file(filename, &file));
	auto fos = std::make_shared<io::FileOutputStream>(file.get());
	auto serializer = std::make_unique<io::JsonSerializer>();
	serializer->attach(fos);
	serializer->write(sparse_features);

	std::unique_ptr<io::RandomAccessFile> raf;
	ASSERT_FALSE(fs->new_random_access_file(filename, &raf));
	auto fis = std::make_shared<io::FileInputStream>(raf.get());
	auto deserializer = std::make_unique<io::JsonDeserializer>();
	deserializer->attach(fis);
	auto sparse_features_loaded = deserializer->read_object();

	ASSERT_FALSE(fs->delete_file(filename));

	SGMatrix<int32_t> data_loaded =
		sparse_features_loaded->as<SparseFeatures<int32_t>>()
			->get_full_feature_matrix();
	//data_loaded.display_matrix();

	EXPECT_TRUE(data_loaded.equals(data));
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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), data.num_cols);

	EXPECT_EQ(features->get_sparse_feature_vector(0).num_feat_entries, 1);
	EXPECT_EQ(features->get_sparse_feature_vector(1).num_feat_entries, 2);
	EXPECT_EQ(features->get_sparse_feature_vector(2).num_feat_entries, 2);

	EXPECT_EQ(features->get_sparse_feature_vector(0).features[0].entry, data(1,0));
	EXPECT_EQ(features->get_sparse_feature_vector(1).features[0].entry, data(0,1));
	EXPECT_EQ(features->get_sparse_feature_vector(2).features[0].entry, data(0,2));


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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

	SGVector<index_t> subset_idx(2);
	subset_idx[0]=2;
	subset_idx[1]=0;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);
	EXPECT_EQ(features->get_sparse_feature_vector(0).features[0].entry, data(0,2));
	EXPECT_EQ(features->get_sparse_feature_vector(0).features[1].entry, data(1,2));
	EXPECT_EQ(features->get_sparse_feature_vector(1).features[0].entry, data(1,0));


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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

	SGVector<index_t> subset_idx(3);
	subset_idx[0]=0;
	subset_idx[1]=1;
	subset_idx[2]=2;

	features->add_subset(subset_idx);

	EXPECT_EQ(features->get_num_features(), data.num_rows);
	EXPECT_EQ(features->get_num_vectors(), subset_idx.vlen);

	SGMatrix<int32_t> mat=features->get_full_feature_matrix();
	EXPECT_TRUE(mat.equals(data));


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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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

	auto features=std::make_shared<SparseFeatures<int32_t>>(data);

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


}
