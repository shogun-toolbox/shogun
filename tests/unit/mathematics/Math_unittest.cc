/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein
 * Written (W) 2013 Soumyajit De
 */

#include <lib/common.h>
#include <lib/SGVector.h>
#include <mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(CMath, qsort_test)
{
	// testing qsort on list of zero elements
	CMath::qsort((int32_t *)NULL, 0);

	// testing qsort on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	CMath::qsort(v1, 1);
	SG_FREE(v1);
}

TEST(CMath, qsort_ptr_test)
{
	// testing qsort on list of zero pointers
	CMath::qsort((int32_t **)NULL, 0);

	// testing qsort on list of one pointer
	int32_t ** v1 = SG_CALLOC(int32_t *, 1);
	CMath::qsort(v1, 1);
	SG_FREE(v1);
}

TEST(CMath, qsort_index_test)
{
	// testing qsort_index on list of zero elements
	CMath::qsort_index((int32_t *)NULL, (int32_t *)NULL, 0);

	// testing qsort_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	CMath::qsort_index(v1, i1, 1);
	SG_FREE(v1);
	SG_FREE(i1);
}

TEST(CMath, qsort_backward_index_test)
{
	// testing qsort_backward_index on list of zero elements
	CMath::qsort_backward_index((int32_t *)NULL, (int32_t *)NULL, 0);

	// testing qsort_backward_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	CMath::qsort_backward_index(v1, i1, 1);
	SG_FREE(v1);
	SG_FREE(i1);
}

TEST(CMath, parallel_qsort_index_test)
{
	// testing parallel_qsort_index on list of zero elements
	CMath::parallel_qsort_index((int32_t *)NULL, (int32_t *)NULL, 0, 8);

	// testing parallel_qsort_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	CMath::parallel_qsort_index(v1, i1, 1, 8);
	SG_FREE(v1);
	SG_FREE(i1);
}

TEST(CMath, float64_tests)
{
	// round, ceil, floor
	EXPECT_NEAR(CMath::round(7.5), 8.0, 1E-15);
	EXPECT_NEAR(CMath::round(7.5-1E-15), 7.0, 1E-15);
	EXPECT_NEAR(CMath::floor(8-1E-15), 7.0, 1E-15);
	EXPECT_NEAR(CMath::ceil(7+1E-15), 8.0, 1E-15);

	float64_t a=5.78123516567856743364;
	// x^2, x^(1/2)
	EXPECT_NEAR(CMath::sq(a), 33.42268004087848964900, 1E-15);
	EXPECT_NEAR(CMath::sqrt(33.42268004087848964900), a, 1E-15);
	EXPECT_NEAR(CMath::pow(a, 2), 33.42268004087848964900, 1E-15);
	EXPECT_NEAR(CMath::pow(33.42268004087848964900, 0.5), a, 1E-15);

	// e^x, log_{b}(x)
	EXPECT_NEAR(CMath::exp(a), 324.15933372813628920994, 1E-15);
	EXPECT_NEAR(CMath::log2(a), 2.53137775864743908016, 1E-15);
	EXPECT_NEAR(CMath::log10(a), 0.76202063570953693095, 1E-15);

	// exp and log identities
	EXPECT_NEAR(CMath::log(CMath::exp(a)), a, 1E-15);
	EXPECT_NEAR(CMath::exp(CMath::log(a)), a, 1E-15);

	// trigonometric functions
	EXPECT_NEAR(CMath::sin(a), -0.48113603605414501097, 1E-15);
	EXPECT_NEAR(CMath::sinh(a), 162.07812441272406545067, 1E-13);
	EXPECT_NEAR(CMath::asin(a-5.0), 0.89664205584230471935, 1E-15);
	EXPECT_NEAR(CMath::cos(a), 0.87664594609802681813, 1E-15);
	EXPECT_NEAR(CMath::cosh(a), 162.08120931541219533756, 1E-15);
	EXPECT_NEAR(CMath::acos(a-5.0), 0.67415427095259194967, 1E-15);
	EXPECT_NEAR(CMath::tan(a), -0.54883734784344084812, 1E-15);
	EXPECT_NEAR(CMath::tanh(a), 0.99998096693194016282, 1E-15);
	EXPECT_NEAR(CMath::atan(a), 1.39951769800256187182, 1E-15);

	// trigonometric identities
	EXPECT_NEAR(CMath::sq(CMath::sin(a))+CMath::sq(CMath::cos(a)),
		1.0, 1E-15);
	EXPECT_NEAR(CMath::sq(1.0/CMath::cos(a))-CMath::sq(CMath::tan(a)),
		1.0, 1E-15);
	EXPECT_NEAR(CMath::sq(1.0/CMath::sin(a))-CMath::sq(1.0/CMath::tan(a)),
		1.0, 1E-15);

	// misc
	SGVector<float64_t> vec(10);
	for (index_t i=0; i<10; ++i)
	{
		vec[i]=i%2==0 ? i : 0.0;
	}
	EXPECT_EQ(CMath::get_num_nonzero(vec.vector, 10), 4);
}

TEST(CMath, linspace_test)
{
	// Number of points used to divide the interval
	int32_t n = 100;
	// Start and end of the interval
	float64_t start = 0.0, end = 1.0;

	SGVector<float64_t> vec(100);
	CMath::linspace(vec.vector, start, end, n);

	// The first and last elements are tested outside the loop, because
	// linspace sets them directly using the arguments
	EXPECT_EQ(vec[0], start);
	EXPECT_EQ(vec[n-1], end);

	float64_t val = start;
	for (index_t i = 1; i < n-1; ++i)
	{
		val += (end-start)/(n-1);
		EXPECT_EQ(vec[i], val);
	}
}

TEST(CMath, log_sum_exp)
{
	SGVector<float64_t> values(3);
	values.range_fill();
	EXPECT_NEAR(CMath::log_sum_exp(values), 2.4076059644443801, 1e-15);
}

TEST(CMath, log_mean_exp)
{
	SGVector<float64_t> values(3);
	values.range_fill();
	EXPECT_NEAR(CMath::log_mean_exp(values), 1.3089936757762706, 1e-15);
}
