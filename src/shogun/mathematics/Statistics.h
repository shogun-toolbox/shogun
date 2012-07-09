/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * ALGLIB Copyright 1984, 1987, 1995, 2000 by Stephen L. Moshier under GPL2+
 * http://www.alglib.net/
 * See method comments which functions are taken from ALGLIB (with adjustments
 * for shogun)
 */

#ifndef __STATISTICS_H_
#define __STATISTICS_H_

#include <math.h>
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
template<class T> class SGMatrix;

/** @brief Class that contains certain functions related to statistics, such as
 * the student's t distribution.
 */
class CStatistics: public CSGObject
{

public:

	/** Calculates mean of given values
	 *
	 * @param values vector of values
	 * @return variance of given values
	 */
	static float64_t mean(SGVector<float64_t> values);

	/** Calculates variance of given values
	 *
	 * @param values vector of values
	 * @return variance of given values
	 */
	static float64_t variance(SGVector<float64_t> values);

#ifdef HAVE_LAPACK
	/** Computes the empirical estimate of the covariance matrix of the given
	 * data which is organized as num_cols variables with num_rows observations.
	 *
	 * TODO latex
	 *
	 * Data is centered before matrix is computed. May be done in place.
	 * In this case, the observation matrix is changed (centered).
	 *
	 * @param observations data matrix organized as one variable per column
	 * @param in_place optional, if set to true, observations matrix will be
	 * centered, if false, a copy will be created an centered.
	 * @return covariance matrix empirical estimate
	 */
	static SGMatrix<float64_t> covariance_matrix(
			SGMatrix<float64_t> observations, bool in_place=false);
#endif //HAVE_LAPACK

	/** Calculates standard deviation of given values
	 *
	 * @param values vector of values
	 * @return standard deviation of given values
	 */
	static float64_t std_deviation(SGVector<float64_t> values);

	/** Calculates the sample mean of a given set of samples and also computes
	 * the confidence interval for the actual mean for a given p-value,
	 * assuming that the actual variance and mean are unknown (These are
	 * estimated by the samples). Based on Student's t-distribution.
	 *
	 * Only for normally distributed data
	 *
	 * @param values vector of values that are used for calculations
	 * @param alpha actual mean lies in confidence interval with (1-alpha)*100%
	 * @param conf_int_low lower confidence interval border is written here
	 * @param conf_int_up upper confidence interval border is written here
	 * @return sample mean
	 *
	 */
	static float64_t confidence_intervals_mean(SGVector<float64_t> values,
			float64_t alpha, float64_t& conf_int_low, float64_t& conf_int_up);

	/** Functional inverse of Student's t distribution
	 *
	 * Given probability p, finds the argument t such that stdtr(k,t)
	 * is equal to p.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t inverse_student_t(int32_t k, float64_t p);

	/** Inverse of imcomplete beta integral
	 *
	 * Given y, the function finds x such that
	 *
	 * incbet( a, b, x ) = y .
	 *
	 * The routine performs interval halving or Newton iterations to find the
	 * root of incbet(a,b,x) - y = 0.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t inverse_incomplete_beta(float64_t a, float64_t b,
			float64_t y);

	/** Incomplete beta integral
	 *
	 * Returns incomplete beta integral of the arguments, evaluated
	 * from zero to x.  The function is defined as
	 *                  x
	 *     -            -
	 *    | (a+b)      | |  a-1     b-1
	 *  -----------    |   t   (1-t)   dt.
	 *   -     -     | |
	 *  | (a) | (b)   -
	 *                 0
	 *
	 * The domain of definition is 0 <= x <= 1.  In this
	 * implementation a and b are restricted to positive values.
	 * The integral from x to 1 may be obtained by the symmetry
	 * relation
	 *
	 *    1 - incbet( a, b, x )  =  incbet( b, a, 1-x ).
	 *
	 * The integral is evaluated by a continued fraction expansion
	 * or, when b*x is small, by a power series.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t incomplete_beta(float64_t a, float64_t b, float64_t x);

	/** Inverse of Normal distribution function
	 *
	 * Returns the argument, x, for which the area under the
	 * Gaussian probability density function (integrated from
	 * minus infinity to x) is equal to y.
	 *
	 *
	 * For small arguments 0 < y < exp(-2), the program computes
	 * z = sqrt( -2.0 * log(y) );  then the approximation is
	 * x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z).
	 * There are two rational functions P/Q, one for 0 < y < exp(-32)
	 * and the other for y up to exp(-2).  For larger arguments,
	 * w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t inverse_normal_distribution(float64_t y0);

	/** @return natural logarithm of the gamma function of input */
	static inline float64_t lgamma(float64_t x)
	{
		return ::lgamma((double) x);
	}

	static inline floatmax_t lgammal(floatmax_t x)
	{
#ifdef HAVE_LGAMMAL
		return ::lgammal((long double) x);
#else
		return ::lgamma((double) x);
#endif // HAVE_LGAMMAL
	}

	/** @return gamma function of input */
	static inline float64_t tgamma(float64_t x)
	{
		return ::tgamma((double) x);
	}

	/** Incomplete gamma integral
	 *
	 * Given p, the function finds x such that
	 *
	 * The function is defined by
	 *
	 *                           x
	 *                            -
	 *                   1       | |  -t  a-1
	 *  igam(a,x)  =   -----     |   e   t   dt.
	 *                  -      | |
	 *                 | (a)    -
	 *                           0
	 *
	 *
	 * In this implementation both arguments must be positive.
	 * The integral is evaluated by either a power series or
	 * continued fraction expansion, depending on the relative
	 * values of a and x.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t incomplete_gamma(float64_t a, float64_t x);

	/** Complemented incomplete gamma integral
	 *
	 * The function is defined by
	 *
	 *
	 * igamc(a,x)   =   1 - igam(a,x)
	 *
	 *                            inf.
	 *                              -
	 *                     1       | |  -t  a-1
	 *               =   -----     |   e   t   dt.
	 *                    -      | |
	 *                   | (a)    -
	 *                             x
	 *
	 *
	 * In this implementation both arguments must be positive.
	 * The integral is evaluated by either a power series or
	 * continued fraction expansion, depending on the relative
	 * values of a and x.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t incomplete_gamma_completed(float64_t a, float64_t x);

	/** Evaluates the CDF of the gamma distribution with given parameters a, b
	 * at x. Based on Wikipedia definition and ALGLIB routines.
	 *
	 * @param x position to evaluate
	 * @param a shape parameter
	 * @param b scale parameter
	 */
	static float64_t gamma_cdf(float64_t x, float64_t a, float64_t b);

	/** Normal distribution function
	 *
	 * Returns the area under the Gaussian probability density
	 * function, integrated from minus infinity to x:
	 *
	 *                            x
	 *                             -
	 *                   1        | |          2
	 *    ndtr(x)  = ---------    |    exp( - t /2 ) dt
	 *               sqrt(2pi)  | |
	 *                           -
	 *                          -inf.
	 *
	 *             =  ( 1 + erf(z) ) / 2
	 *
	 * where z = x/sqrt(2)/std_dev. Computation is via the functions
	 * erf and erfc.
	 *
	 * Taken from ALGLIB under gpl2+
	 * Custom variance added by Heiko Strathmann
	 */
	static float64_t normal_cdf(float64_t x, float64_t std_dev=1);

	/** Error function
	 *
	 * The integral is
	 *
	 *                           x
	 *                            -
	 *                 2         | |          2
	 *   erf(x)  =  --------     |    exp( - t  ) dt.
	 *              sqrt(pi)   | |
	 *                          -
	 *                           0
	 *
	 * For 0 <= |x| < 1, erf(x) = x * P4(x**2)/Q5(x**2); otherwise
	 * erf(x) = 1 - erfc(x).
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t error_function(float64_t x);

	/** Complementary error function
	 *
	 * 1 - erf(x) =
	 *
	 *                           inf.
	 *                             -
	 *                  2         | |          2
	 *   erfc(x)  =  --------     |    exp( - t  ) dt
	 *               sqrt(pi)   | |
	 *                           -
	 *                            x
	 *
	 *
	 * For small x, erfc(x) = 1 - erf(x); otherwise rational
	 * approximations are computed.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t error_function_complement(float64_t x);

	/// returns the mutual information of p which is given in logspace
	/// where p,q are given in logspace
	static float64_t mutual_info(float64_t* p1, float64_t* p2, int32_t len);

	/// returns the relative entropy H(P||Q),
	/// where p,q are given in logspace
	static float64_t relative_entropy(
			float64_t* p, float64_t* q, int32_t len);

	/// returns entropy of p which is given in logspace
	static float64_t entropy(float64_t* p, int32_t len);

	/** fisher's test for multiple 2x3 tables
	 * @param tables
	 */
	static SGVector<float64_t> fishers_exact_test_for_multiple_2x3_tables(SGMatrix<float64_t> tables);

	/** fisher's test for 2x3 table
	 * @param table
	 */
	static float64_t fishers_exact_test_for_2x3_table(SGMatrix<float64_t> table);

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "Statistics";
	}

protected:
	/** Power series for incomplete beta integral.
	 * Use when b*x is small and x not too close to 1.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetaps(float64_t a, float64_t b,
			float64_t x, float64_t maxgam);

	/** Continued fraction expansion #1 for incomplete beta integral
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetafe(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);

	/** Continued fraction expansion #2 for incomplete beta integral
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetafe2(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);

	/** method to make ALGLIB integration easier */
	static inline bool equal(float64_t a, float64_t b) { return a==b; }

	/** method to make ALGLIB integration easier */
	static inline bool not_equal(float64_t a, float64_t b) { return a!=b; }

	/** method to make ALGLIB integration easier */
	static inline bool less(float64_t a, float64_t b) { return a<b; }

	/** method to make ALGLIB integration easier */
	static inline bool less_equal(float64_t a, float64_t b) { return a<=b; }

	/** method to make ALGLIB integration easier */
	static inline bool greater(float64_t a, float64_t b) { return a>b; }

	/** method to make ALGLIB integration easier */
	static inline bool greater_equal(float64_t a, float64_t b) { return a>=b; }
};

}

#endif /* __STATISTICS_H_ */
