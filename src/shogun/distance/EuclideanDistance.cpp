/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Written (W) 2016 Soumyajit De
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CEuclideanDistance::CEuclideanDistance() : CDistance()
{
	register_params();
}

CEuclideanDistance::CEuclideanDistance(CDotFeatures* l, CDotFeatures* r) : CDistance()
{
	register_params();
	init(l, r);
}

CEuclideanDistance::~CEuclideanDistance()
{
	cleanup();
}

bool CEuclideanDistance::init(CFeatures* l, CFeatures* r)
{
	cleanup();

	CDistance::init(l, r);
	REQUIRE(l->has_property(FP_DOT), "Left hand side features must support dot property!\n");
	REQUIRE(r->has_property(FP_DOT), "Right hand side features must support dot property!\n");

	CDotFeatures* casted_l=static_cast<CDotFeatures*>(l);
	CDotFeatures* casted_r=static_cast<CDotFeatures*>(r);

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

void CEuclideanDistance::cleanup()
{
	reset_precompute();
}

float64_t CEuclideanDistance::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=0;
	CDotFeatures* casted_lhs=static_cast<CDotFeatures*>(lhs);
	CDotFeatures* casted_rhs=static_cast<CDotFeatures*>(rhs);

	if (lhs->get_feature_class()==rhs->get_feature_class() || lhs->support_compatible_class())
		result=casted_lhs->dot(idx_a, casted_rhs, idx_b);
	else
		result=casted_rhs->dot(idx_b, casted_lhs, idx_a);

	result=m_lhs_squared_norms[idx_a]+m_rhs_squared_norms[idx_b]-2*result;
	if (disable_sqrt)
		return result;
	return CMath::sqrt(result);
}

void CEuclideanDistance::precompute_lhs()
{
	REQUIRE(lhs, "Left hand side feature cannot be NULL!\n");
	const index_t num_vec=lhs->get_num_vectors();

	if (m_lhs_squared_norms.vlen!=num_vec)
		m_lhs_squared_norms=SGVector<float64_t>(num_vec);

	CDotFeatures* casted_lhs=static_cast<CDotFeatures*>(lhs);
#pragma omp parallel for schedule(static, CPU_CACHE_LINE_SIZE_BYTES)
	for(index_t i =0; i<num_vec; ++i)
		m_lhs_squared_norms[i]=casted_lhs->dot(i, casted_lhs, i);
}

void CEuclideanDistance::precompute_rhs()
{
	REQUIRE(rhs, "Right hand side feature cannot be NULL!\n");
	const index_t num_vec=rhs->get_num_vectors();

	if (m_rhs_squared_norms.vlen!=num_vec)
		m_rhs_squared_norms=SGVector<float64_t>(num_vec);

	CDotFeatures* casted_rhs=static_cast<CDotFeatures*>(rhs);
#pragma omp parallel for schedule(static, CPU_CACHE_LINE_SIZE_BYTES)
	for(index_t i =0; i<num_vec; ++i)
		m_rhs_squared_norms[i]=casted_rhs->dot(i, casted_rhs, i);
}

void CEuclideanDistance::reset_precompute()
{
	m_lhs_squared_norms=SGVector<float64_t>();
	m_rhs_squared_norms=SGVector<float64_t>();
}

CFeatures* CEuclideanDistance::replace_lhs(CFeatures* l)
{
	CFeatures* previous_lhs=CDistance::replace_lhs(l);
	precompute_lhs();
	if (lhs==rhs)
		m_rhs_squared_norms=m_lhs_squared_norms;
	return previous_lhs;
}

CFeatures* CEuclideanDistance::replace_rhs(CFeatures* r)
{
	CFeatures* previous_rhs=CDistance::replace_rhs(r);
	if (lhs==rhs)
		m_rhs_squared_norms=m_lhs_squared_norms;
	else
		precompute_rhs();
	return previous_rhs;
}

void CEuclideanDistance::register_params()
{
	disable_sqrt=false;
	reset_precompute();
	SG_ADD(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.", MS_NOT_AVAILABLE);
	SG_ADD(&m_rhs_squared_norms, "m_rhs_squared_norms", "Squared norms from features of right hand side", MS_NOT_AVAILABLE);
	SG_ADD(&m_lhs_squared_norms, "m_lhs_squared_norms", "Squared norms from features of left hand side", MS_NOT_AVAILABLE);
}

float64_t CEuclideanDistance::distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound)
{
	REQUIRE(lhs->get_feature_class()==C_DENSE,
		"Left hand side (was %s) has to be CDenseFeatures instance!\n", lhs->get_name());
	REQUIRE(rhs->get_feature_class()==C_DENSE,
		"Right hand side (was %s) has to be CDenseFeatures instance!\n", rhs->get_name());

	REQUIRE(lhs->get_feature_type()==F_DREAL,
		"Left hand side (was %s) has to be of double type!\n", lhs->get_name());
	REQUIRE(rhs->get_feature_type()==F_DREAL,
		"Right hand side (was %s) has to be double type!\n", rhs->get_name());

	CDenseFeatures<float64_t>* casted_lhs=static_cast<CDenseFeatures<float64_t>*>(lhs);
	CDenseFeatures<float64_t>* casted_rhs=static_cast<CDenseFeatures<float64_t>*>(rhs);

	upper_bound*=upper_bound;

	SGVector<float64_t> avec=casted_lhs->get_feature_vector(idx_a);
	SGVector<float64_t> bvec=casted_rhs->get_feature_vector(idx_b);

	REQUIRE(avec.vlen==bvec.vlen, "The vector lengths are not equal (%d vs %d)!\n", avec.vlen, bvec.vlen);

	float64_t result=0;
	for (int32_t i=0; i<avec.vlen; i++)
	{
		result+=CMath::sq(avec[i]-bvec[i]);
		if (result>upper_bound)
			break;
	}

	if (!disable_sqrt)
		result=CMath::sqrt(result);

	return result;
}
