/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Soumyajit De, Chiyuan Zhang, Viktor Gal,
 *          Bjoern Esser, Soeren Sonnenburg
 */

#include <shogun/lib/common.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

EuclideanDistance::EuclideanDistance() : Distance()
{
	register_params();
}

EuclideanDistance::EuclideanDistance(std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r) : Distance()
{
	register_params();
	init(l, r);
}

EuclideanDistance::~EuclideanDistance()
{
	cleanup();
}

bool EuclideanDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	cleanup();

	Distance::init(l, r);
	REQUIRE(l->has_property(FP_DOT), "Left hand side features must support dot property!\n");
	REQUIRE(r->has_property(FP_DOT), "Right hand side features must support dot property!\n");

	auto casted_l=std::dynamic_pointer_cast<DotFeatures>(l);
	auto casted_r=std::dynamic_pointer_cast<DotFeatures>(r);

	REQUIRE(casted_l->get_dim_feature_space()==casted_r->get_dim_feature_space(),
		"Number of dimension mismatch (l:%d vs. r:%d)!\n",
		casted_l->get_dim_feature_space(),casted_r->get_dim_feature_space());

	precompute_lhs();
	if (lhs==rhs)
		m_rhs_squared_norms=m_lhs_squared_norms;
	else
		precompute_rhs();

	return true;
}

void EuclideanDistance::cleanup()
{
	reset_precompute();
}

float64_t EuclideanDistance::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=0;
	auto casted_lhs=std::dynamic_pointer_cast<DotFeatures>(lhs);
	auto casted_rhs=std::dynamic_pointer_cast<DotFeatures>(rhs);

	if (lhs->get_feature_class()==rhs->get_feature_class() || lhs->support_compatible_class())
		result=casted_lhs->dot(idx_a, casted_rhs, idx_b);
	else
		result=casted_rhs->dot(idx_b, casted_lhs, idx_a);

	result=m_lhs_squared_norms[idx_a]+m_rhs_squared_norms[idx_b]-2*result;
	if (disable_sqrt)
		return result;
	return std::sqrt(result);
}

void EuclideanDistance::precompute_lhs()
{
	REQUIRE(lhs, "Left hand side feature cannot be NULL!\n");
	const index_t num_vec=lhs->get_num_vectors();

	if (m_lhs_squared_norms.vlen!=num_vec)
		m_lhs_squared_norms=SGVector<float64_t>(num_vec);

	auto casted_lhs=std::dynamic_pointer_cast<DotFeatures>(lhs);
#pragma omp parallel for schedule(static, CPU_CACHE_LINE_SIZE_BYTES)
	for(index_t i =0; i<num_vec; ++i)
		m_lhs_squared_norms[i]=casted_lhs->dot(i, casted_lhs, i);
}

void EuclideanDistance::precompute_rhs()
{
	REQUIRE(rhs, "Right hand side feature cannot be NULL!\n");
	const index_t num_vec=rhs->get_num_vectors();

	if (m_rhs_squared_norms.vlen!=num_vec)
		m_rhs_squared_norms=SGVector<float64_t>(num_vec);

	auto casted_rhs=std::dynamic_pointer_cast<DotFeatures>(rhs);
#pragma omp parallel for schedule(static, CPU_CACHE_LINE_SIZE_BYTES)
	for(index_t i =0; i<num_vec; ++i)
		m_rhs_squared_norms[i]=casted_rhs->dot(i, casted_rhs, i);
}

void EuclideanDistance::reset_precompute()
{
	m_lhs_squared_norms=SGVector<float64_t>();
	m_rhs_squared_norms=SGVector<float64_t>();
}

std::shared_ptr<Features> EuclideanDistance::replace_lhs(std::shared_ptr<Features> l)
{
	auto previous_lhs=Distance::replace_lhs(l);
	precompute_lhs();
	if (lhs==rhs)
		m_rhs_squared_norms=m_lhs_squared_norms;
	return previous_lhs;
}

std::shared_ptr<Features> EuclideanDistance::replace_rhs(std::shared_ptr<Features> r)
{
	auto previous_rhs=Distance::replace_rhs(r);
	if (lhs==rhs)
		m_rhs_squared_norms=m_lhs_squared_norms;
	else
		precompute_rhs();
	return previous_rhs;
}

void EuclideanDistance::register_params()
{
	disable_sqrt=false;
	reset_precompute();
	SG_ADD(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
	SG_ADD(&m_rhs_squared_norms, "m_rhs_squared_norms", "Squared norms from features of right hand side");
	SG_ADD(&m_lhs_squared_norms, "m_lhs_squared_norms", "Squared norms from features of left hand side");
}

float64_t EuclideanDistance::distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound)
{
	REQUIRE(lhs->get_feature_class()==C_DENSE,
		"Left hand side (was %s) has to be DenseFeatures instance!\n", lhs->get_name());
	REQUIRE(rhs->get_feature_class()==C_DENSE,
		"Right hand side (was %s) has to be DenseFeatures instance!\n", rhs->get_name());

	REQUIRE(lhs->get_feature_type()==F_DREAL,
		"Left hand side (was %s) has to be of double type!\n", lhs->get_name());
	REQUIRE(rhs->get_feature_type()==F_DREAL,
		"Right hand side (was %s) has to be double type!\n", rhs->get_name());

	auto casted_lhs=std::dynamic_pointer_cast<DenseFeatures<float64_t>>(lhs);
	auto casted_rhs=std::dynamic_pointer_cast<DenseFeatures<float64_t>>(rhs);

	upper_bound*=upper_bound;

	SGVector<float64_t> avec=casted_lhs->get_feature_vector(idx_a);
	SGVector<float64_t> bvec=casted_rhs->get_feature_vector(idx_b);

	REQUIRE(avec.vlen==bvec.vlen, "The vector lengths are not equal (%d vs %d)!\n", avec.vlen, bvec.vlen);

	float64_t result=0;
	for (int32_t i=0; i<avec.vlen; i++)
	{
		result+=Math::sq(avec[i]-bvec[i]);
		if (result>upper_bound)
			break;
	}

	if (!disable_sqrt)
		result = std::sqrt(result);

	return result;
}
