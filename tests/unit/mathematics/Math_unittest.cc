/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
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
	SGVector<float64_t> sg_vec = CMath::linspace_vec(start, end, n);

	// The first and last elements are tested outside the loop, because
	// linspace sets them directly using the arguments
	EXPECT_EQ(vec[0], start);
	EXPECT_EQ(vec[n-1], end);

	// test for CMath::linspace_vec which returns a vector
	EXPECT_EQ(sg_vec[0], start);
	EXPECT_EQ(sg_vec[n-1], end);

	float64_t val = start;
	for (index_t i = 1; i < n-1; ++i)
	{
		val += (end-start)/(n-1);
		EXPECT_EQ(vec[i], val);
		EXPECT_EQ(sg_vec[i], val);
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

TEST(CMath, strtofloat)
{
	float32_t float_result = 0;
	EXPECT_TRUE(CMath::strtof("nan", &float_result));
	EXPECT_TRUE(CMath::is_nan(float_result));

	EXPECT_TRUE(CMath::strtof("inf", &float_result));
	EXPECT_TRUE(CMath::is_infinity(float_result));

	EXPECT_TRUE(CMath::strtof("-inf", &float_result));
	EXPECT_DOUBLE_EQ(-CMath::INFTY, float_result);

	EXPECT_TRUE(CMath::strtof("1.2345", &float_result));
	EXPECT_FLOAT_EQ(1.2345, float_result);
}

TEST(CMath, strtodouble)
{
	float64_t double_result = 0;
	EXPECT_TRUE(CMath::strtod("nan", &double_result));
	EXPECT_TRUE(CMath::is_nan(double_result));

	EXPECT_TRUE(CMath::strtod("inf", &double_result));
	EXPECT_TRUE(CMath::is_infinity(double_result));

	EXPECT_TRUE(CMath::strtod("-inf", &double_result));
	EXPECT_DOUBLE_EQ(-CMath::INFTY, double_result);

	EXPECT_TRUE(CMath::strtod("1.234567890123", &double_result));
	EXPECT_DOUBLE_EQ(1.234567890123, double_result);
}

TEST(CMath, strtolongdouble)
{
	floatmax_t long_double_result = 0;
	EXPECT_TRUE(CMath::strtold("nan", &long_double_result));
	EXPECT_TRUE(CMath::is_nan(long_double_result));

	EXPECT_TRUE(CMath::strtold("inf", &long_double_result));
	EXPECT_TRUE(CMath::is_infinity(long_double_result));

	EXPECT_TRUE(CMath::strtold("-inf", &long_double_result));
	EXPECT_DOUBLE_EQ(-CMath::INFTY, long_double_result);

	EXPECT_TRUE(CMath::strtold("1.234567890123", &long_double_result));
	EXPECT_DOUBLE_EQ(1.234567890123, long_double_result);
}

TEST(CMath, fequals_regular_large_numbers)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(1000000.0, 1000000.0, eps));
	EXPECT_TRUE(CMath::fequals<floatmax_t>(1000000.0, 1000000.0, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(10001.0, 10000.0, eps));
	EXPECT_FALSE(CMath::fequals<floatmax_t>(10000.0, 10001.0, eps));
}

TEST(CMath, fequals_negative_large_numbers)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(-100000.0, -100000.0, eps));
	EXPECT_TRUE(CMath::fequals<floatmax_t>(-1000001.0, -1000001.0, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(1000001.0, 1000000.0, eps));
	EXPECT_FALSE(CMath::fequals<floatmax_t>(1000000.0, 1000001.0, eps));
}

TEST(CMath, fequals_numbers_around_1)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(1.0000001, 1.0000002, eps));
	EXPECT_TRUE(CMath::fequals<floatmax_t>(1.0000002, 1.0000001, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(1.0002, 1.0001, eps));
	EXPECT_FALSE(CMath::fequals<floatmax_t>(1.0002, 1.0001, eps));
}

TEST(CMath, fequals_numbers_around_minus_1)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(-1.0000001, -1.0000002, eps));
	EXPECT_TRUE(CMath::fequals<floatmax_t>(-1.0000002, -1.0000001, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-1.0002, -1.0001, eps));
	EXPECT_FALSE(CMath::fequals<floatmax_t>(-1.0002, -1.0001, eps));
}

TEST(CMath, fequals_small_pos_numbers)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(0.000000001000001, 0.000000001000002, eps));
	EXPECT_TRUE(CMath::fequals<floatmax_t>(0.000000001000002, 0.000000001000001, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.000000000001002, 0.000000000001001, eps));
	EXPECT_FALSE(CMath::fequals<floatmax_t>(0.000000000001001, 0.000000000001002, eps));
}

TEST(CMath, fequals_small_neg_numbers)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(-0.000000001000001, -0.000000001000002, eps));
	EXPECT_TRUE(CMath::fequals<floatmax_t>(-0.000000001000002, -0.000000001000001, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-0.000000000001002, -0.000000000001001, eps));
	EXPECT_FALSE(CMath::fequals<floatmax_t>(-0.000000000001001, -0.000000000001002, eps));
}

TEST(CMath, fequals_zero)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(0.0, 0.0, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(0.0, -0.0, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(-0.0, -0.0, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.00000001, 0.0, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.0, 0.00000001, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-0.00000001, 0.0, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.0, -0.00000001, eps));
	
	EXPECT_TRUE(CMath::fequals<float32_t>(0.0, 1e-40, 0.01));
	EXPECT_TRUE(CMath::fequals<float32_t>(1e-40, 0.0, 0.01));
	EXPECT_TRUE(CMath::fequals<float32_t>(0.0, 1e-40, 0.01));
	EXPECT_TRUE(CMath::fequals<float32_t>(1e-40, 0.0, 0.01));
	
	EXPECT_FALSE(CMath::fequals<float64_t>(0.0, 1e-40, 0.01));
	EXPECT_FALSE(CMath::fequals<float64_t>(1e-40, 0.0, 0.01));
	EXPECT_FALSE(CMath::fequals<float64_t>(1e-40, 0.0, 0.000001));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.0, 1e-40, 0.000001));

	EXPECT_FALSE(CMath::fequals<float64_t>(0.0, -1e-40, 0.1));
	EXPECT_FALSE(CMath::fequals<float64_t>(-1e-40, 0.0, 0.1));
	EXPECT_FALSE(CMath::fequals<float64_t>(-1e-40, 0.0, 0.00000001));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.0, -1e-40, 0.00000001));
}

TEST(CMath, fequals_inf)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(CMath::INFTY, CMath::INFTY, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(-CMath::INFTY, -CMath::INFTY, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-CMath::INFTY, CMath::INFTY, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::INFTY, CMath::F_MAX_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-CMath::INFTY, -CMath::F_MAX_VAL64, eps));
}

TEST(CMath, fequals_nan)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, 0.0f, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-0.0f, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, -0.0f, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.0f, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, CMath::INFTY, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::INFTY, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, -CMath::INFTY, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-CMath::INFTY, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, CMath::F_MAX_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::F_MAX_VAL64, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, -CMath::F_MAX_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-CMath::F_MAX_VAL64, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, CMath::F_MIN_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::F_MIN_VAL64, CMath::NOT_A_NUMBER, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::NOT_A_NUMBER, -CMath::F_MIN_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-CMath::F_MIN_VAL64, CMath::NOT_A_NUMBER, eps));
}

TEST(CMath, fequals_opposite_sign)
{
	float64_t eps = 0.00001;
	
	EXPECT_FALSE(CMath::fequals<float64_t>(1.000000001f, -1.0f, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-1.0f, 1.000000001f, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-1.000000001f, 1.0f, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(1.0f, -1.000000001f, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(10 * CMath::F_MIN_VAL64, 10 * -CMath::F_MIN_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float32_t>(10000 * CMath::F_MIN_VAL32, 10000 * -CMath::F_MIN_VAL32, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(10000 * CMath::F_MIN_VAL64, 10000 * -CMath::F_MIN_VAL64, eps));
}

TEST(CMath, fequals_close_to_zero)
{
	float64_t eps = 0.00001;
	
	EXPECT_TRUE(CMath::fequals<float64_t>(CMath::F_MIN_VAL64, -CMath::F_MIN_VAL64, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(-CMath::F_MIN_VAL64, CMath::F_MIN_VAL64, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(CMath::F_MIN_VAL64, 0, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(0, CMath::F_MIN_VAL64, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(-CMath::F_MIN_VAL64, 0, eps));
	EXPECT_TRUE(CMath::fequals<float64_t>(0, -CMath::F_MIN_VAL64, eps));

	EXPECT_FALSE(CMath::fequals<float64_t>(0.000000001f, -CMath::F_MIN_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(0.000000001f, CMath::F_MIN_VAL64, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(CMath::F_MIN_VAL64, 0.000000001f, eps));
	EXPECT_FALSE(CMath::fequals<float64_t>(-CMath::F_MIN_VAL64, 0.000000001f, eps));
}

TEST(CMath, get_abs_tolerance)
{
	EXPECT_EQ(CMath::get_abs_tolerance(0.0, 0.01), 0.01);
	EXPECT_NEAR(CMath::get_abs_tolerance(-0.01, 0.01), 0.0001, 1E-15);
	EXPECT_NEAR(CMath::get_abs_tolerance(-9.5367431640625e-7, 0.01), 9.5367431640625e-9, 1E-15);
	EXPECT_NEAR(CMath::get_abs_tolerance(9.5367431640625e-7, 0.01), 9.5367431640625e-9, 1E-15);
	EXPECT_EQ(CMath::get_abs_tolerance(-CMath::F_MIN_VAL64, 0.01), CMath::F_MIN_VAL64);
	EXPECT_EQ(CMath::get_abs_tolerance(CMath::F_MIN_VAL64, 0.01), CMath::F_MIN_VAL64);

}

TEST(CMath, permute)
{
	SGVector<int32_t> v(4);
	v.range_fill(0);
	CMath::init_random(2);
	CMath::permute(v);

	EXPECT_EQ(v[0], 2);
	EXPECT_EQ(v[1], 1);
	EXPECT_EQ(v[2], 3);
	EXPECT_EQ(v[3], 0);
}

TEST(CMath, permute_with_random)
{
	SGVector<int32_t> v(4);
	v.range_fill(0);
	CRandom* random = new CRandom(2);
	CMath::permute(v, random);
	SG_UNREF(random);

	EXPECT_EQ(v[0], 2);
	EXPECT_EQ(v[1], 1);
	EXPECT_EQ(v[2], 3);
	EXPECT_EQ(v[3], 0);
}

TEST(CMath,misc)
{
	CMath::init_random(17);
	SGVector<float64_t> a(10);
	a.random(-1024.0, 1024.0);

	/* test, min, max */
	int arg_max = 0;
	float64_t min = 1025, max = -1025;
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		if (a[i] > max)
		{
			max = a[i];
			arg_max=i;
		}
		if (a[i] < min)
			min = a[i];
	}

	EXPECT_EQ(min, CMath::min(a.vector,a.vlen));
	EXPECT_EQ(max, CMath::max(a.vector,a.vlen));
	EXPECT_EQ(arg_max, CMath::arg_max(a.vector,1, a.vlen));
}

TEST(CMath,vector_qsort_test)
{
	SGVector<index_t> v(4);
	v[0]=12;
	v[1]=1;
	v[2]=7;
	v[3]=9;

	CMath::qsort(v);

	EXPECT_EQ(v.vlen, 4);
	EXPECT_EQ(v[0], 1);
	EXPECT_EQ(v[1], 7);
	EXPECT_EQ(v[2], 9);
	EXPECT_EQ(v[3], 12);
}

TEST(CMath,is_sorted)
{
	SGVector<index_t> v(4);
	v[0]=12;
	v[1]=1;
	v[2]=7;
	v[3]=9;

	EXPECT_EQ(CMath::is_sorted(v), false);
	CMath::qsort(v);

	EXPECT_EQ(CMath::is_sorted(v), true);
}

TEST(CMath,is_sorted_0)
{
	SGVector<index_t> v(0);

	EXPECT_EQ(CMath::is_sorted(v), true);
	CMath::qsort(v);

	EXPECT_EQ(CMath::is_sorted(v), true);
}

TEST(CMath,is_sorted_1)
{
	SGVector<index_t> v(1);
	v[0]=12;

	EXPECT_EQ(CMath::is_sorted(v), true);
	CMath::qsort(v);

	EXPECT_EQ(CMath::is_sorted(v), true);
}

TEST(CMath,is_sorted_2)
{
	SGVector<index_t> v(2);
	v[0]=12;
	v[1]=1;

	EXPECT_EQ(CMath::is_sorted(v), false);
	CMath::qsort(v);

	EXPECT_EQ(CMath::is_sorted(v), true);
}

TEST(CMath, dot)
{
	CMath::init_random(17);
	SGVector<float64_t> a(10);
	a.random(0.0, 1024.0);
	float64_t dot_val = 0.0;

	for (int32_t i = 0; i < a.vlen; ++i)
		dot_val += a[i]*a[i];

	float64_t sgdot_val = CMath::dot(a.vector,a.vector, a.vlen);
	EXPECT_NEAR(dot_val, sgdot_val, 1e-9);
}