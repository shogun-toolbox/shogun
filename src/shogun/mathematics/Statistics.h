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

	/** @return object name */
	inline virtual const char* get_name() const { return "CStatistics"; }

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
	 * Taken from ALGOLIB under gpl2+
	 */
	static float64_t inverse_student_t_distribution(int32_t k, float64_t p);

	/** Inverse of imcomplete beta integral
	 *
	 * Given y, the function finds x such that
	 *
	 * incbet( a, b, x ) = y .
	 *
	 * The routine performs interval halving or Newton iterations to find the
	 * root of incbet(a,b,x) - y = 0.
	 *
	 * Taken from ALGOLIB under gpl2+
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
	 * Taken from ALGOLIB under gpl2+
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
	 * Taken from ALGOLIB under gpl2+
	 */
	static float64_t inverse_normal_distribution(float64_t y0);

	/** Incomplete gamma integral
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
	 * Taken from ALGOLIB under gpl2+
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
	 * Taken from ALGOLIB under gpl2+
	 */
	static float64_t incomplete_gamma_completed(float64_t a, float64_t x);

	protected:
	/** Power series for incomplete beta integral.
	 * Use when b*x is small and x not too close to 1.
	 *
	 * Taken from ALGOLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetaps(float64_t a, float64_t b,
			float64_t x, float64_t maxgam);

	/** Continued fraction expansion #1 for incomplete beta integral
	 *
	 * Taken from ALGOLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetafe(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);

	/** Continued fraction expansion #2 for incomplete beta integral
	 *
	 * Taken from ALGOLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetafe2(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);

	static inline bool equal(float64_t a, float64_t b) { return a==b; }
	static inline bool not_equal(float64_t a, float64_t b) { return a!=b; }
	static inline bool less(float64_t a, float64_t b) { return a<b; }
	static inline bool less_equal(float64_t a, float64_t b) { return a<=b; }
	static inline bool greater(float64_t a, float64_t b) { return a>b; }
	static inline bool greater_equal(float64_t a, float64_t b) { return a>=b; }
};

}

#endif /* __STATISTICS_H_ */
