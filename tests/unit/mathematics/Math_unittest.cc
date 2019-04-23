/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sanuj Sharma, Viktor Gal, Fernando Iglesias, Heiko Strathmann, 
 *          syashakash, Soumyajit De, Bjoern Esser, Soeren Sonnenburg, Wu Lin, 
 *          Grigorii Guz, Albert, Akash Shivram, Thoralf Klein, Shubham Shukla
 */
#include <gtest/gtest.h>

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/RandomNamespace.h>

#include <random>

using namespace shogun;

TEST(Math, qsort_test)
{
	// testing qsort on list of zero elements
	Math::qsort((int32_t *)NULL, 0);

	// testing qsort on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	Math::qsort(v1, 1);
	SG_FREE(v1);
}

TEST(Math, qsort_ptr_test)
{
	// testing qsort on list of zero pointers
	Math::qsort((int32_t **)NULL, 0);

	// testing qsort on list of one pointer
	int32_t ** v1 = SG_CALLOC(int32_t *, 1);
	Math::qsort(v1, 1);
	SG_FREE(v1);
}

TEST(Math, qsort_index_test)
{
	// testing qsort_index on list of zero elements
	Math::qsort_index((int32_t *)NULL, (int32_t *)NULL, 0);

	// testing qsort_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	Math::qsort_index(v1, i1, 1);
	SG_FREE(v1);
	SG_FREE(i1);
}

TEST(Math, qsort_backward_index_test)
{
	// testing qsort_backward_index on list of zero elements
	Math::qsort_backward_index((int32_t *)NULL, (int32_t *)NULL, 0);

	// testing qsort_backward_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	Math::qsort_backward_index(v1, i1, 1);
	SG_FREE(v1);
	SG_FREE(i1);
}

#ifdef HAVE_PTHREAD
TEST(Math, parallel_qsort_index_test)
{
	// testing parallel_qsort_index on list of zero elements
	Math::parallel_qsort_index((int32_t *)NULL, (int32_t *)NULL, 0, 8);

	// testing parallel_qsort_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	Math::parallel_qsort_index(v1, i1, 1, 8);
	SG_FREE(v1);
	SG_FREE(i1);
}
#endif

TEST(Math, float64_tests)
{
	// round, ceil, floor
	EXPECT_NEAR(Math::round(7.5), 8.0, 1E-15);
	EXPECT_NEAR(Math::round(7.5-1E-15), 7.0, 1E-15);
	EXPECT_NEAR(Math::floor(8-1E-15), 7.0, 1E-15);
	EXPECT_NEAR(std::ceil(7 + 1E-15), 8.0, 1E-15);

	float64_t a=5.78123516567856743364;
	// x^2, x^(1/2)
	EXPECT_NEAR(Math::sq(a), 33.42268004087848964900, 1E-15);
	EXPECT_NEAR(std::sqrt(33.42268004087848964900), a, 1E-15);
	EXPECT_NEAR(Math::pow(a, 2), 33.42268004087848964900, 1E-15);
	EXPECT_NEAR(Math::pow(33.42268004087848964900, 0.5), a, 1E-15);

	// e^x, log_{b}(x)
	EXPECT_NEAR(std::exp(a), 324.15933372813628920994, 1E-15);
	EXPECT_NEAR(Math::log2(a), 2.53137775864743908016, 1E-15);
	EXPECT_NEAR(Math::log10(a), 0.76202063570953693095, 1E-15);

	// exp and log identities
	EXPECT_NEAR(std::log(std::exp(a)), a, 1E-15);
	EXPECT_NEAR(std::exp(std::log(a)), a, 1E-15);

	// trigonometric functions
	EXPECT_NEAR(std::sin(a), -0.48113603605414501097, 1E-15);
	EXPECT_NEAR(std::sinh(a), 162.07812441272406545067, 1E-13);
	EXPECT_NEAR(std::asin(a - 5.0), 0.89664205584230471935, 1E-15);
	EXPECT_NEAR(std::cos(a), 0.87664594609802681813, 1E-15);
	EXPECT_NEAR(std::cosh(a), 162.08120931541219533756, 1E-15);
	EXPECT_NEAR(std::acos(a - 5.0), 0.67415427095259194967, 1E-15);
	EXPECT_NEAR(std::tan(a), -0.54883734784344084812, 1E-15);
	EXPECT_NEAR(std::tanh(a), 0.99998096693194016282, 1E-15);
	EXPECT_NEAR(std::atan(a), 1.39951769800256187182, 1E-15);

	// trigonometric identities
	EXPECT_NEAR(Math::sq(std::sin(a)) + Math::sq(std::cos(a)), 1.0, 1E-15);
	EXPECT_NEAR(
	    Math::sq(1.0 / std::cos(a)) - Math::sq(std::tan(a)), 1.0, 1E-15);
	EXPECT_NEAR(
	    Math::sq(1.0 / std::sin(a)) - Math::sq(1.0 / std::tan(a)), 1.0,
	    1E-15);

	// misc
	SGVector<float64_t> vec(10);
	for (index_t i=0; i<10; ++i)
	{
		vec[i]=i%2==0 ? i : 0.0;
	}
	EXPECT_EQ(Math::get_num_nonzero(vec.vector, 10), 4);
}

TEST(Math, linspace_test)
{
	// Number of points used to divide the interval
	int32_t n = 100;
	// Start and end of the interval
	float64_t start = 0.0, end = 1.0;

	SGVector<float64_t> vec(100);
	Math::linspace(vec.vector, start, end, n);
	SGVector<float64_t> sg_vec = Math::linspace_vec(start, end, n);

	// The first and last elements are tested outside the loop, because
	// linspace sets them directly using the arguments
	EXPECT_EQ(vec[0], start);
	EXPECT_EQ(vec[n-1], end);

	// test for Math::linspace_vec which returns a vector
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

TEST(Math, log_sum_exp)
{
	SGVector<float64_t> values(3);
	values.range_fill();
	EXPECT_NEAR(Math::log_sum_exp(values), 2.4076059644443801, 1e-15);
}

TEST(Math, log_mean_exp)
{
	SGVector<float64_t> values(3);
	values.range_fill();
	EXPECT_NEAR(Math::log_mean_exp(values), 1.3089936757762706, 1e-15);
}

TEST(Math, strtofloat)
{
	float32_t float_result = 0;
	EXPECT_TRUE(Math::strtof("nan", &float_result));
	EXPECT_TRUE(Math::is_nan(float_result));

	EXPECT_TRUE(Math::strtof("inf", &float_result));
	EXPECT_TRUE(std::isinf(float_result));

	EXPECT_TRUE(Math::strtof("-inf", &float_result));
	EXPECT_DOUBLE_EQ(-Math::INFTY, float_result);

	EXPECT_TRUE(Math::strtof("1.2345", &float_result));
	EXPECT_FLOAT_EQ(1.2345, float_result);
}

TEST(Math, strtodouble)
{
	float64_t double_result = 0;
	EXPECT_TRUE(Math::strtod("nan", &double_result));
	EXPECT_TRUE(Math::is_nan(double_result));

	EXPECT_TRUE(Math::strtod("inf", &double_result));
	EXPECT_TRUE(std::isinf(double_result));

	EXPECT_TRUE(Math::strtod("-inf", &double_result));
	EXPECT_DOUBLE_EQ(-Math::INFTY, double_result);

	EXPECT_TRUE(Math::strtod("1.234567890123", &double_result));
	EXPECT_DOUBLE_EQ(1.234567890123, double_result);
}

TEST(Math, strtolongdouble)
{
	floatmax_t long_double_result = 0;
	EXPECT_TRUE(Math::strtold("nan", &long_double_result));
	EXPECT_TRUE(Math::is_nan(long_double_result));

	EXPECT_TRUE(Math::strtold("inf", &long_double_result));
	EXPECT_TRUE(std::isinf(long_double_result));

	EXPECT_TRUE(Math::strtold("-inf", &long_double_result));
	EXPECT_DOUBLE_EQ(-Math::INFTY, long_double_result);

	EXPECT_TRUE(Math::strtold("1.234567890123", &long_double_result));
	EXPECT_DOUBLE_EQ(1.234567890123, long_double_result);
}

TEST(Math, fequals_regular_large_numbers)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(1000000.0, 1000000.0, eps));
	EXPECT_TRUE(Math::fequals<floatmax_t>(1000000.0, 1000000.0, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(10001.0, 10000.0, eps));
	EXPECT_FALSE(Math::fequals<floatmax_t>(10000.0, 10001.0, eps));
}

TEST(Math, fequals_negative_large_numbers)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(-100000.0, -100000.0, eps));
	EXPECT_TRUE(Math::fequals<floatmax_t>(-1000001.0, -1000001.0, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(1000001.0, 1000000.0, eps));
	EXPECT_FALSE(Math::fequals<floatmax_t>(1000000.0, 1000001.0, eps));
}

TEST(Math, fequals_numbers_around_1)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(1.0000001, 1.0000002, eps));
	EXPECT_TRUE(Math::fequals<floatmax_t>(1.0000002, 1.0000001, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(1.0002, 1.0001, eps));
	EXPECT_FALSE(Math::fequals<floatmax_t>(1.0002, 1.0001, eps));
}

TEST(Math, fequals_numbers_around_minus_1)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(-1.0000001, -1.0000002, eps));
	EXPECT_TRUE(Math::fequals<floatmax_t>(-1.0000002, -1.0000001, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-1.0002, -1.0001, eps));
	EXPECT_FALSE(Math::fequals<floatmax_t>(-1.0002, -1.0001, eps));
}

TEST(Math, fequals_small_pos_numbers)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(0.000000001000001, 0.000000001000002, eps));
	EXPECT_TRUE(Math::fequals<floatmax_t>(0.000000001000002, 0.000000001000001, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(0.000000000001002, 0.000000000001001, eps));
	EXPECT_FALSE(Math::fequals<floatmax_t>(0.000000000001001, 0.000000000001002, eps));
}

TEST(Math, fequals_small_neg_numbers)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(-0.000000001000001, -0.000000001000002, eps));
	EXPECT_TRUE(Math::fequals<floatmax_t>(-0.000000001000002, -0.000000001000001, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-0.000000000001002, -0.000000000001001, eps));
	EXPECT_FALSE(Math::fequals<floatmax_t>(-0.000000000001001, -0.000000000001002, eps));
}

TEST(Math, fequals_zero)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(0.0, 0.0, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(0.0, -0.0, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(-0.0, -0.0, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(0.00000001, 0.0, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(0.0, 0.00000001, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-0.00000001, 0.0, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(0.0, -0.00000001, eps));

	EXPECT_TRUE(Math::fequals<float32_t>(0.0, 1e-40, 0.01));
	EXPECT_TRUE(Math::fequals<float32_t>(1e-40, 0.0, 0.01));
	EXPECT_TRUE(Math::fequals<float32_t>(0.0, 1e-40, 0.01));
	EXPECT_TRUE(Math::fequals<float32_t>(1e-40, 0.0, 0.01));

	EXPECT_FALSE(Math::fequals<float64_t>(0.0, 1e-40, 0.01));
	EXPECT_FALSE(Math::fequals<float64_t>(1e-40, 0.0, 0.01));
	EXPECT_FALSE(Math::fequals<float64_t>(1e-40, 0.0, 0.000001));
	EXPECT_FALSE(Math::fequals<float64_t>(0.0, 1e-40, 0.000001));

	EXPECT_FALSE(Math::fequals<float64_t>(0.0, -1e-40, 0.1));
	EXPECT_FALSE(Math::fequals<float64_t>(-1e-40, 0.0, 0.1));
	EXPECT_FALSE(Math::fequals<float64_t>(-1e-40, 0.0, 0.00000001));
	EXPECT_FALSE(Math::fequals<float64_t>(0.0, -1e-40, 0.00000001));
}

TEST(Math, fequals_inf)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(Math::INFTY, Math::INFTY, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(-Math::INFTY, -Math::INFTY, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-Math::INFTY, Math::INFTY, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::INFTY, Math::F_MAX_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-Math::INFTY, -Math::F_MAX_VAL64, eps));
}

TEST(Math, fequals_nan)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, 0.0f, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-0.0f, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, -0.0f, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(0.0f, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, Math::INFTY, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::INFTY, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, -Math::INFTY, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-Math::INFTY, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, Math::F_MAX_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::F_MAX_VAL64, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, -Math::F_MAX_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-Math::F_MAX_VAL64, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, Math::F_MIN_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::F_MIN_VAL64, Math::NOT_A_NUMBER, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::NOT_A_NUMBER, -Math::F_MIN_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-Math::F_MIN_VAL64, Math::NOT_A_NUMBER, eps));
}

TEST(Math, fequals_opposite_sign)
{
	float64_t eps = 0.00001;

	EXPECT_FALSE(Math::fequals<float64_t>(1.000000001f, -1.0f, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-1.0f, 1.000000001f, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-1.000000001f, 1.0f, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(1.0f, -1.000000001f, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(10 * Math::F_MIN_VAL64, 10 * -Math::F_MIN_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float32_t>(10000 * Math::F_MIN_VAL32, 10000 * -Math::F_MIN_VAL32, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(10000 * Math::F_MIN_VAL64, 10000 * -Math::F_MIN_VAL64, eps));
}

TEST(Math, fequals_close_to_zero)
{
	float64_t eps = 0.00001;

	EXPECT_TRUE(Math::fequals<float64_t>(Math::F_MIN_VAL64, -Math::F_MIN_VAL64, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(-Math::F_MIN_VAL64, Math::F_MIN_VAL64, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(Math::F_MIN_VAL64, 0, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(0, Math::F_MIN_VAL64, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(-Math::F_MIN_VAL64, 0, eps));
	EXPECT_TRUE(Math::fequals<float64_t>(0, -Math::F_MIN_VAL64, eps));

	EXPECT_FALSE(Math::fequals<float64_t>(0.000000001f, -Math::F_MIN_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(0.000000001f, Math::F_MIN_VAL64, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(Math::F_MIN_VAL64, 0.000000001f, eps));
	EXPECT_FALSE(Math::fequals<float64_t>(-Math::F_MIN_VAL64, 0.000000001f, eps));
}

TEST(Math, get_abs_tolerance)
{
	EXPECT_EQ(Math::get_abs_tolerance(0.0, 0.01), 0.01);
	EXPECT_NEAR(Math::get_abs_tolerance(-0.01, 0.01), 0.0001, 1E-15);
	EXPECT_NEAR(Math::get_abs_tolerance(-9.5367431640625e-7, 0.01), 9.5367431640625e-9, 1E-15);
	EXPECT_NEAR(Math::get_abs_tolerance(9.5367431640625e-7, 0.01), 9.5367431640625e-9, 1E-15);
	EXPECT_EQ(Math::get_abs_tolerance(-Math::F_MIN_VAL64, 0.01), Math::F_MIN_VAL64);
	EXPECT_EQ(Math::get_abs_tolerance(Math::F_MIN_VAL64, 0.01), Math::F_MIN_VAL64);

}

TEST(Math,misc)
{
	std::mt19937_64 prng(17);
	SGVector<float64_t> a(10);
	random::fill_array(a, -1024.0, 1024.0, prng);

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

	EXPECT_EQ(min, Math::min(a.vector,a.vlen));
	EXPECT_EQ(max, Math::max(a.vector,a.vlen));
	EXPECT_EQ(arg_max, Math::arg_max(a.vector,1, a.vlen));
}

TEST(Math,vector_qsort_test)
{
	SGVector<index_t> v(4);
	v[0]=12;
	v[1]=1;
	v[2]=7;
	v[3]=9;

	Math::qsort(v);

	EXPECT_EQ(v.vlen, 4);
	EXPECT_EQ(v[0], 1);
	EXPECT_EQ(v[1], 7);
	EXPECT_EQ(v[2], 9);
	EXPECT_EQ(v[3], 12);
}

TEST(Math,is_sorted)
{
	SGVector<index_t> v(4);
	v[0]=12;
	v[1]=1;
	v[2]=7;
	v[3]=9;

	EXPECT_EQ(Math::is_sorted(v), false);
	Math::qsort(v);

	EXPECT_EQ(Math::is_sorted(v), true);
}

TEST(Math,is_sorted_0)
{
	SGVector<index_t> v(0);

	EXPECT_EQ(Math::is_sorted(v), true);
	Math::qsort(v);

	EXPECT_EQ(Math::is_sorted(v), true);
}

TEST(Math,is_sorted_1)
{
	SGVector<index_t> v(1);
	v[0]=12;

	EXPECT_EQ(Math::is_sorted(v), true);
	Math::qsort(v);

	EXPECT_EQ(Math::is_sorted(v), true);
}

TEST(Math,is_sorted_2)
{
	SGVector<index_t> v(2);
	v[0]=12;
	v[1]=1;

	EXPECT_EQ(Math::is_sorted(v), false);
	Math::qsort(v);

	EXPECT_EQ(Math::is_sorted(v), true);
}

TEST(Math, gcd)
{
	EXPECT_EQ(Math::gcd(12,8), 4);
	EXPECT_EQ(Math::gcd(18,27), 9);
	EXPECT_EQ(Math::gcd(1,1), 1);
	EXPECT_EQ(Math::gcd(1,2), 1);
	EXPECT_EQ(Math::gcd(1,0), 1);
	EXPECT_EQ(Math::gcd(0,1), 1);
	EXPECT_THROW(Math::gcd(0,0), ShogunException);
	EXPECT_THROW(Math::gcd(1,-1), ShogunException);
	EXPECT_THROW(Math::gcd(-1,1), ShogunException);

}
