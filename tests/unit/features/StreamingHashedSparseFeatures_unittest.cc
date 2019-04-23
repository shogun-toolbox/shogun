/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Thoralf Klein, Bjoern Esser
 */

#include <gtest/gtest.h>
#include <shogun/lib/Hash.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/streaming/StreamingHashedSparseFeatures.h>

using namespace shogun;

TEST(StreamingHashedSparseFeaturesTest, dot)
{
	index_t n=3;
	index_t dim=10;

	SGMatrix<float64_t> data(dim,n);
	for (index_t i=0; i<n; i++)
	{
		for (index_t j=0; j<dim; j++)
			data(j,i) = j + i * dim;
	}

	int32_t hashing_dim = 8;
	auto d_feats = std::make_shared<SparseFeatures<float64_t>>(data);
	auto h_feats =
		std::make_shared<StreamingHashedSparseFeatures<float64_t>>(d_feats, hashing_dim);

	h_feats->start_parser();
	index_t i;
	for (i=0; i<n && h_feats->get_next_example(); i++)
	{
		SGVector<uint32_t> tmp(hashing_dim);
		SGVector<uint32_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			if (data(j,i)==0)
				continue;
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dot(h_feats);
		EXPECT_EQ(feat_dot, dot_product);
		h_feats->release_example();
	}
	h_feats->end_parser();

	EXPECT_EQ(i, n);


}

TEST(StreamingHashedSparseFeaturesTest, dense_dot)
{
	index_t n=3;
	index_t dim=10;

	SGMatrix<float64_t> data(dim,n);
	for (index_t i=0; i<n; i++)
	{
		for (index_t j=0; j<dim; j++)
			data(j,i) = j + i * dim;
	}

	int32_t hashing_dim = 8;

	auto d_feats = std::make_shared<SparseFeatures<float64_t>>(data);
	auto h_feats =
		std::make_shared<StreamingHashedSparseFeatures<float64_t>>(d_feats, hashing_dim);

	h_feats->start_parser();
	for (index_t i=0; i<n && h_feats->get_next_example(); i++)
	{
		SGVector<float32_t> tmp(hashing_dim);
		SGVector<float32_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			if (data(j,i)==0)
				continue;
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dense_dot(tmp.vector, tmp.vlen);
		EXPECT_EQ(feat_dot, dot_product);

		h_feats->release_example();
	}
	h_feats->end_parser();


}

TEST(StreamingHashedSparseFeaturesTest, add_to_dense)
{
	index_t n=3;
	index_t dim=10;

	SGMatrix<float64_t> data(dim,n);
	for (index_t i=0; i<n; i++)
	{
		for (index_t j=0; j<dim; j++)
			data(j,i) = j + i * dim;
	}

	int32_t hashing_dim = 8;
	auto d_feats = std::make_shared<SparseFeatures<float64_t>>(data);
	auto h_feats =
		std::make_shared<StreamingHashedSparseFeatures<float64_t>>(d_feats, hashing_dim);

	h_feats->start_parser();
	for (index_t i=0; i<n && h_feats->get_next_example(); i++)
	{
		SGVector<float32_t> tmp(hashing_dim);
		SGVector<float32_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			if (data(j,i)==0)
				continue;
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		SGVector<float64_t> tmp2(hashing_dim);
		for (index_t j=0; j<hashing_dim; j++)
			tmp2[j] = 3 * tmp[j];

		h_feats->add_to_dense_vec(2, tmp.vector, tmp.vlen);
		for (index_t j=0; j<hashing_dim; j++)
			EXPECT_EQ(tmp2[j], tmp[j]);

		h_feats->release_example();
	}
	h_feats->end_parser();


}
