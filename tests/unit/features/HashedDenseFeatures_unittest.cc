/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Thoralf Klein, Sergey Lisitsyn
 */

#include <gtest/gtest.h>
#include <shogun/lib/Hash.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/hashed/HashedDenseFeatures.h>
#include <shogun/mathematics/UniformIntDistribution.h>

#include <random>

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
	auto h_feats = std::make_shared<HashedDenseFeatures<float64_t>>(data, hashing_dim);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dot(i, h_feats, i);
		EXPECT_EQ(feat_dot, dot_product);
	}


}

TEST(HashedDenseFeaturesTest, quadratic_dot)
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
	auto h_feats = std::make_shared<HashedDenseFeatures<float64_t>>(data, hashing_dim, true);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		for (index_t j=0; j<dim; j++)
		{
			for (index_t k=j; k<dim; k++)
			{
				if (k!=j)
				{
					uint32_t hash_j = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
					uint32_t hash_k = Hash::MurmurHash3((uint8_t* ) &k, sizeof (index_t), k);
					uint32_t hash = (hash_j ^ hash_k) % hashing_dim;
					tmp[hash] += data(j,i) * data(k,i);
				}
				else
				{
					index_t n_idx = j * dim + k;
					uint32_t hash = Hash::MurmurHash3((uint8_t* ) &n_idx, sizeof (index_t), n_idx);
					tmp[hash % hashing_dim] += data(j,i) * data(k,i);
				}
			}
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dot(i, h_feats, i);
		EXPECT_EQ(feat_dot, dot_product);
	}


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
	auto h_feats = std::make_shared<HashedDenseFeatures<float64_t>>(data, hashing_dim);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dot(i, tmp);
		EXPECT_EQ(feat_dot, dot_product);
	}


}

TEST(HashedDenseFeaturesTest, quadratic_dense_dot)
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
	auto h_feats = std::make_shared<HashedDenseFeatures<float64_t>>(data, hashing_dim, true);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		for (index_t j=0; j<dim; j++)
		{
			for (index_t k=j; k<dim; k++)
			{
				if (k!=j)
				{
					uint32_t hash_j = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
					uint32_t hash_k = Hash::MurmurHash3((uint8_t* ) &k, sizeof (index_t), k);
					uint32_t hash = (hash_j ^ hash_k) % hashing_dim;
					tmp[hash] += data(j,i) * data(k,i);
				}
				else
				{
					index_t n_idx = j * dim + k;
					uint32_t hash = Hash::MurmurHash3((uint8_t* ) &n_idx, sizeof (index_t), n_idx);
					tmp[hash % hashing_dim] += data(j,i) * data(k,i);
				}
			}
		}

		float64_t dot_product = 0;
		for (index_t j=0; j<hashing_dim; j++)
			dot_product += tmp[j] * tmp[j];

		float64_t feat_dot = h_feats->dot(i, tmp);
		EXPECT_EQ(feat_dot, dot_product);
	}


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
	auto h_feats = std::make_shared<HashedDenseFeatures<float64_t>>(data, hashing_dim);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<n; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		SGVector<float64_t> tmp2(hashing_dim);
		for (index_t j=0; j<hashing_dim; j++)
			tmp2[j] = 3 * tmp[j];

		h_feats->add_to_dense_vec(2, i, tmp.vector, tmp.vlen);
		for (index_t j=0; j<hashing_dim; j++)
			EXPECT_EQ(tmp2[j], tmp[j]);
	}


}


TEST(HashedDenseFeaturesTest, quadratic_add_to_dense)
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
	auto h_feats = std::make_shared<HashedDenseFeatures<float64_t>>(data, hashing_dim, true);
	EXPECT_EQ(h_feats->get_num_vectors(), n);

	for (index_t i=0; i<3; i++)
	{
		SGVector<float64_t> tmp(hashing_dim);
		SGVector<float64_t>::fill_vector(tmp, hashing_dim, 0);
		for (index_t j=0; j<dim; j++)
		{
			uint32_t hash = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
			hash = hash % hashing_dim;
			tmp[hash] += data(j,i);
		}

		for (index_t j=0; j<dim; j++)
		{
			for (index_t k=j; k<dim; k++)
			{
				if (k!=j)
				{
					uint32_t hash_j = Hash::MurmurHash3((uint8_t* ) &j, sizeof (index_t), j);
					uint32_t hash_k = Hash::MurmurHash3((uint8_t* ) &k, sizeof (index_t), k);
					uint32_t hash = (hash_j ^ hash_k) % hashing_dim;
					tmp[hash] += data(j,i) * data(k,i);
				}
				else
				{
					index_t n_idx = j * dim + k;
					uint32_t hash = Hash::MurmurHash3((uint8_t* ) &n_idx, sizeof (index_t), n_idx);
					tmp[hash % hashing_dim] += data(j,i) * data(k,i);
				}
			}
		}

		SGVector<float64_t> tmp2(hashing_dim);
		for (index_t j=0; j<hashing_dim; j++)
			tmp2[j] = 3 * tmp[j];

		h_feats->add_to_dense_vec(2, i, tmp.vector, tmp.vlen);
		for (index_t j=0; j<hashing_dim; j++)
			EXPECT_EQ(tmp2[j], tmp[j]);
	}


}

TEST(HashedDenseFeaturesTest, dense_comparison)
{
	int32_t seed = 12;
	index_t n=3;
	index_t dim=10;

	SGMatrix<float64_t> data(dim,n);
	for (index_t i=0; i<n; i++)
	{
		for (index_t j=0; j<dim; j++)
			data(j,i) = j + i * dim;
	}

	int32_t hashing_dim = 300;
	auto h_feats = std::make_shared<HashedDenseFeatures<float64_t>>(data, hashing_dim);
	auto d_feats = std::make_shared<DenseFeatures<float64_t>>(data);

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	SGVector<float64_t> dense_vec(hashing_dim);
	for (index_t i=0; i<hashing_dim; i++)
		dense_vec[i] = uniform_int_dist(prng, {-hashing_dim, hashing_dim});

	for (index_t i=0; i<n; i++)
		EXPECT_EQ(h_feats->dot(i, h_feats, i), d_feats->dot(i, d_feats, i));



}
