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

#include <shogun/base/SGObject.h>

namespace shogun
{

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

	/** Calculates standard deviation of given values
	 *
	 * @param values vector of values
	 * @return standard deviation of given values
	 */
	static float64_t std_deviation(SGVector<float64_t> values);

	/** Calculates the sample mean of a given set of samples and also computes
	 * the confidence interval for the actual mean for a given p-value,
	 * asuming that the actual variance and mean are unknown (These are
	 * estimated by the samples)
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

	/** Student's t distribution
	 * Computes the integral from minus infinity to t of the Student
	 * For t < -2, this is the method of computation.  For higher t,
	 * a direct method is derived from integration by parts.
	 * Since the function is symmetric about t=0, the area under the
	 * right tail of the density is found by calling the function
	 * with -t instead of t.
	 * Taken from ALGLIB under GPL2+

	 * @param k degrees of freedom
	 * @param t integral is computed from minus infinity to t
	 * @return described integral
	 */
	static float64_t student_t_distribution(int32_t k, float64_t t);

	/** Functional inverse of Student's t distribution
	 * Given probability p, finds the argument t such that stdtr(k,t) is equal
	 * to p.
	 *
	 * Taken from ALGLIB under GPL2+
	 *
	 */
	static float64_t inverse_student_t_distribution(int32_t k, float64_t p);

	/** Incomplete beta integral
	 * Returns incomplete beta integral of the arguments, evaluated
	 * from zero to x.
	 * The domain of definition is 0 <= x <= 1.  In this
	 * implementation a and b are restricted to positive values.
	 * The integral is evaluated by a continued fraction expansion
	 * or, when b*x is small, by a power series.
	 *
	 * Taken from ALGLIB under GPL2+
	 */
	static float64_t incomplete_beta(float64_t a, float64_t b, float64_t x);

	/** Inverse of imcomplete beta integral
	 * Given y, the function finds x such that
	 * inverse_incomplete_beta(a, b, x) = y .
	 * The routine performs interval halving or Newton iterations to find the
	 * root of inverse_incomplete_beta(a, b, x)-y=0.
	 *
	 * Taken from ALGLIB under GPL2+
	 */
	static float64_t inverse_incomplete_beta(float64_t a, float64_t b,
			float64_t y);

	/** Inverse of Normal distribution function
	 * Returns the argument, x, for which the area under the Gaussian
	 * probability density function (integrated from minus infinity to x) is
	 * equal to y.
	 *
	 * For small arguments 0 < y < exp(-2), the program computes
	 * z=sqrt(-2.0*log(y))
	 * then the approximation is
	 * x=z-log(z)/z-(1/z)P(1/z)/Q(1/z).
	 * There are two rational functions P/Q, one for 0 < y < exp(-32)
	 * and the other for y up to exp(-2).  For larger arguments,
	 * w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
	 *
	 * Taken from ALGLIB under GPL2+
	 */
	static float64_t inverse_normal_distribution(float64_t y0);

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "Statistics";
	}

protected:
	/**
	 * Power series for incomplete beta integral.
	 * Use when b*x is small and x not too close to 1.
	 *
	 * Taken from ALGLIB under GPL2+
	 */
	static float64_t ibetaf_incomplete_beta_ps(float64_t a, float64_t b,
			float64_t x, float64_t maxgam);

	/** Continued fraction expansion #1 for incomplete beta integral
	 *
	 * Taken from ALGLIB under GPL2+
	 */
	static float64_t ibetaf_incomplete_beta_fe(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);

	/**Continued fraction expansion #2 for incomplete beta integral
	 *
	 * Taken from ALGLIB under GPL2+
	 */
	static float64_t ibetaf_incomplete_beta_fe2(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);
};

}

#endif /* __STATISTICS_H_ */
