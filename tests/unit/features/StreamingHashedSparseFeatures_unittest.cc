/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/base/init.h>
#include <shogun/lib/Hash.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/streaming/StreamingHashedSparseFeatures.h>
#include <gtest/gtest.h>

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
	CSparseFeatures<float64_t>* d_feats = new CSparseFeatures<float64_t>(data);
	CStreamingHashedSparseFeatures<float64_t>* h_feats =
		new CStreamingHashedSparseFeatures<float64_t>(d_feats, hashing_dim);

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
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
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

	SG_UNREF(h_feats);
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

	CSparseFeatures<float64_t>* d_feats = new CSparseFeatures<float64_t>(data);
	CStreamingHashedSparseFeatures<float64_t>* h_feats =
		new CStreamingHashedSparseFeatures<float64_t>(d_feats, hashing_dim);

	h_feats->start_parser();
	for (index_t i=0; i<n && h_feats->get_next_example(); i++)
	{
		SGVector<float32_t> tmp(hashing_dim);
		SGVector<float32_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			if (data(j,i)==0)
				continue;
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
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

	SG_UNREF(h_feats);
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
	CSparseFeatures<float64_t>* d_feats = new CSparseFeatures<float64_t>(data);
	CStreamingHashedSparseFeatures<float64_t>* h_feats =
		new CStreamingHashedSparseFeatures<float64_t>(d_feats, hashing_dim);

	h_feats->start_parser();
	for (index_t i=0; i<n && h_feats->get_next_example(); i++)
	{
		SGVector<float32_t> tmp(hashing_dim);
		SGVector<float32_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			if (data(j,i)==0)
				continue;
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
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

	SG_UNREF(h_feats);
}
