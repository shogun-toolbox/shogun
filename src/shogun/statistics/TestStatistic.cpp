/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/TestStatistic.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CTestStatistic::CTestStatistic() : CSGObject()
{
	init();
}

CTestStatistic::~CTestStatistic()
{

}

void CTestStatistic::init()
{
	SG_ADD(&m_num_permutation_iterations, "num_permutation_iterations",
			"Number of permutation iterations for sampling null",
			MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_null_approximation_method,
			"null_approximation_method",
			"Method for approximating null distribution",
			MS_NOT_AVAILABLE);

	m_num_permutation_iterations=250;
	m_null_approximation_method=PERMUTATION;
}

void CTestStatistic::set_null_approximation_method(
		ENullApproximationMethod null_approximation_method)
{
	m_null_approximation_method=null_approximation_method;
}

void CTestStatistic::set_num_permutation_iterations(index_t
		num_permutation_iterations)
{
	m_num_permutation_iterations=num_permutation_iterations;
}

float64_t CTestStatistic::perform_test()
{
	/* baseline method here is simply to compute statistic and p-value
	 * separately */
	float64_t statistic=compute_statistic();
	return compute_p_value(statistic);
}

bool CTestStatistic::perform_test(float64_t alpha)
{
	float64_t p_value=perform_test();
	return p_value<alpha;
}
