/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/lib/Hash.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/HashedDenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(HashedDenseFeaturesTest, dot)
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
	CHashedDenseFeatures<float64_t>* h_feats = new CHashedDenseFeatures<float64_t>(data, hashing_dim);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<uint32_t> tmp(hashing_dim);
		SGVector<uint32_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &data(j,i), sizeof (float64_t), j);
			hash = hash % hashing_dim;
			tmp[hash]++;
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dot(i, h_feats, i);
		EXPECT_EQ(feat_dot, dot_product);
	}

	SG_UNREF(h_feats);
}

TEST(HashedDenseFeaturesTest, dense_dot)
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
	CHashedDenseFeatures<float64_t>* h_feats = new CHashedDenseFeatures<float64_t>(data, hashing_dim);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &data(j,i), sizeof (float64_t), j);
			hash = hash % hashing_dim;
			tmp[hash]++;
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dense_dot(i, tmp.vector, tmp.vlen);
		EXPECT_EQ(feat_dot, dot_product);
	}

	SG_UNREF(h_feats);
}

TEST(HashedDenseFeaturesTest, add_to_dense)
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
	CHashedDenseFeatures<float64_t>* h_feats = new CHashedDenseFeatures<float64_t>(data, hashing_dim);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &data(j,i), sizeof (float64_t), j);
			hash = hash % hashing_dim;
			tmp[hash]++;
		}

		SGVector<float64_t> tmp2(hashing_dim);
		for (index_t j=0; j<hashing_dim; j++)
			tmp2[j] = 3 * tmp[j];

		h_feats->add_to_dense_vec(2, i, tmp.vector, tmp.vlen);
		for (index_t j=0; j<hashing_dim; j++)
			EXPECT_EQ(tmp2[j], tmp[j]);
	}

	SG_UNREF(h_feats);
}

TEST(HashedDenseFeaturesTest, static_quadratic_test)
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

	for (index_t i=0; i<n; i++)
	{
		float64_t* vec = data.get_column_vector(i);
		SGVector<uint32_t> hashed_vec(hashing_dim);
		SGVector<uint32_t>::fill_vector(hashed_vec.vector, hashed_vec.vlen, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &vec[j], sizeof (float64_t), j);
			hashed_vec[hash % hashing_dim]++;
		}

		for (index_t j=0; j<dim; j++)
			for (index_t k=j; k<dim; k++)
			{
				float64_t value = vec[j] * vec[k];
				uint8_t hash = CHash::MurmurHash3((uint8_t* ) &value, sizeof (float64_t), k+j);
				hashed_vec[hash % hashing_dim]++;
			}

		SGSparseVector<uint32_t> hashed_vec_2 = CHashedDenseFeatures<float64_t>::hash_vector(
				SGVector<float64_t>(vec, dim, false), hashing_dim, true);
		
		int32_t dense_idx = 0;
		for (index_t j=0; j<hashed_vec_2.num_feat_entries; j++)
		{
			while (dense_idx < hashed_vec_2.features[j].feat_index)
				EXPECT_EQ(hashed_vec[dense_idx++], 0);
			EXPECT_EQ(hashed_vec_2.features[j].entry, hashed_vec[dense_idx++]);
		}
	}
}
