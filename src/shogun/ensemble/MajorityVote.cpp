/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <ensemble/MajorityVote.h>
#include <lib/SGMatrix.h>

using namespace shogun;

CMajorityVote::CMajorityVote()
	: CWeightedMajorityVote()
{

}

CMajorityVote::~CMajorityVote()
{

}

SGVector<float64_t> CMajorityVote::combine(const SGMatrix<float64_t>& ensemble_result) const
{
	m_weights.resize_vector(ensemble_result.num_cols);
	m_weights.set_const(1.0);

	SGVector<float64_t> combined_result = CWeightedMajorityVote::combine(ensemble_result);

	return combined_result;
}

float64_t CMajorityVote::combine(const SGVector<float64_t>& ensemble_result) const
{
	m_weights.resize_vector(ensemble_result.vlen);
	m_weights.set_const(1.0);

	return weighted_combine(ensemble_result);
}
