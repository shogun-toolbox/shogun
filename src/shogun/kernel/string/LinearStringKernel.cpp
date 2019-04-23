/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sanuj Sharma, Sergey Lisitsyn, Viktor Gal
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/kernel/string/LinearStringKernel.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

LinearStringKernel::LinearStringKernel()
: StringKernel<char>(0)
{
}

LinearStringKernel::LinearStringKernel(
	std::shared_ptr<StringFeatures<char>> l, std::shared_ptr<StringFeatures<char>> r)
: StringKernel<char>(0)
{
	init(l, r);
}

LinearStringKernel::~LinearStringKernel()
{
	cleanup();
}

bool LinearStringKernel::init(std::shared_ptr<Features >l, std::shared_ptr<Features >r)
{
	StringKernel<char>::init(l, r);
	return init_normalizer();
}

void LinearStringKernel::cleanup()
{
	delete_optimization();

	Kernel::cleanup();
}

void LinearStringKernel::clear_normal()
{
	memset(m_normal.vector, 0, lhs->get_num_vectors()*sizeof(float64_t));
}

void LinearStringKernel::add_to_normal(int32_t idx, float64_t weight)
{
	int32_t vlen;
	bool vfree;
	char* vec = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx, vlen, vfree);

	for (int32_t i=0; i<vlen; i++)
		m_normal.vector[i] += weight*normalizer->normalize_lhs(vec[i], idx);

	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(vec, idx, vfree);
}

float64_t LinearStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = lhs->as<StringFeatures<char>>()->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = rhs->as<StringFeatures<char>>()->get_feature_vector(idx_b, blen, free_bvec);
	ASSERT(alen==blen)
	SGVector<char> a_wrap(avec, alen, false);
	SGVector<char> b_wrap(bvec, blen, false);
	float64_t result = linalg::dot(a_wrap, b_wrap);
	lhs->as<StringFeatures<char>>()->free_feature_vector(avec, idx_a, free_avec);
	rhs->as<StringFeatures<char>>()->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

bool LinearStringKernel::init_optimization(
	int32_t num_suppvec, int32_t *sv_idx, float64_t *alphas)
{
	auto sf_lhs = lhs->as<StringFeatures<char>>();
	int32_t num_feat = sf_lhs->get_max_vector_length();
	ASSERT(num_feat)

	m_normal = SGVector<float64_t>(num_feat);
	ASSERT(m_normal.vector)
	clear_normal();

	for (int32_t i = 0; i<num_suppvec; i++)
	{
		int32_t alen;
		bool free_avec;
		char *avec = sf_lhs->get_feature_vector(sv_idx[i], alen, free_avec);
		ASSERT(avec)

		for (int32_t j = 0; j<num_feat; j++)
		{
			m_normal.vector[j] += alphas[i]*
				normalizer->normalize_lhs(((float64_t) avec[j]), sv_idx[i]);
		}
		sf_lhs->free_feature_vector(avec, sv_idx[i], free_avec);
	}
	set_is_initialized(true);
	return true;
}

bool LinearStringKernel::delete_optimization()
{
	m_normal = SGVector<float64_t>();
	set_is_initialized(false);
	return true;
}

float64_t LinearStringKernel::compute_optimized(int32_t idx_b)
{
	int32_t blen;
	bool free_bvec;
	char* bvec = lhs->as<StringFeatures<char>>()->get_feature_vector(idx_b, blen, free_bvec);
	float64_t dot = 0.0;
	for (auto i = 0; m_normal.vlen; ++i)
		dot += m_normal[i]*(float64_t)bvec[i];
	float64_t result=normalizer->normalize_rhs(dot, idx_b);
	rhs->as<StringFeatures<char>>()->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}
