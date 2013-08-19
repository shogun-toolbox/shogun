/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <shogun/ensemble/WeightedMajorityVote.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGMatrix.h>
#include <map>

using namespace shogun;

CWeightedMajorityVote::CWeightedMajorityVote()
	: CCombinationRule()
{
	init();
	register_parameters();
}

CWeightedMajorityVote::CWeightedMajorityVote(SGVector<float64_t>& weights)
	: CCombinationRule()
{
	init();
	register_parameters();
	m_weights = weights;
}

CWeightedMajorityVote::~CWeightedMajorityVote()
{

}

SGVector<float64_t> CWeightedMajorityVote::combine(const SGMatrix<float64_t>& ensemble_result) const
{
	REQUIRE(m_weights.vlen == ensemble_result.num_cols, "The number of results and weights does not match!");
	SGVector<float64_t> mv(ensemble_result.num_rows);
	for (index_t i = 0; i < ensemble_result.num_rows; ++i)
	{
		SGVector<float64_t> rv = ensemble_result.get_row_vector(i);
		mv[i] = combine(rv);
	}

	return mv;
}

float64_t CWeightedMajorityVote::combine(const SGVector<float64_t>& ensemble_result) const
{
	return weighted_combine(ensemble_result);
}

float64_t CWeightedMajorityVote::weighted_combine(const SGVector<float64_t>& ensemble_result) const
{
	REQUIRE(m_weights.vlen == ensemble_result.vlen, "The number of results and weights does not match!");
	std::map<index_t, float64_t> freq;
	std::map<index_t, float64_t>::iterator it;
	index_t max_label = -100;
	float64_t max = CMath::ALMOST_NEG_INFTY;

	for (index_t i = 0; i < ensemble_result.vlen; ++i)
	{
		it = freq.find(ensemble_result[i]);
		if (it == freq.end())
		{
			freq.insert(std::make_pair(ensemble_result[i], m_weights[i]));
			if (max < m_weights[i])
			{
				max_label = ensemble_result[i];
				max = m_weights[i];
			}
		} 
		else
		{
			it->second += m_weights[i];
			if (max < it->second)
			{
				max_label = it->first;
				max = it->second;
			}
		}
	}

	return max_label;
}

void CWeightedMajorityVote::set_weights(SGVector<float64_t>& w)
{
	m_weights = w;
}

SGVector<float64_t> CWeightedMajorityVote::get_weights() const
{
	return m_weights;
}

void CWeightedMajorityVote::init()
{
	m_weights.resize_vector(0);
}

void CWeightedMajorityVote::register_parameters()
{
	SG_ADD(&m_weights, "weights", "Weights for the majority vote", MS_AVAILABLE);
}
