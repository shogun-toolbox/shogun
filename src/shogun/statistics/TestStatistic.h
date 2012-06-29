/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __TESTSTATISTIC_H_
#define __TESTSTATISTIC_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief Test statistic base class. Provides an interface for statistical
 * tests via three methods: compute_statistic(), compute_p_value() and
 * compute_threshold(). The second computes a p-value for the statistic computed
 * by the first method.
 * The p-value represents the position of the statistic in the null-distribution,
 * i.e. the distribution of the statistic population given the null-hypothesis
 * is true. (1-position = p-value).
 * The third method,  compute_threshold(), computes a threshold for a given
 * test level which is needed to reject the null-hypothesis
 *
 * Abstract base class.
 */
class CTestStatistic : public CSGObject
{
	public:
		CTestStatistic() {};

		virtual ~CTestStatistic() {};

		virtual float64_t compute_statistic()
		{
			SG_ERROR("%s::compute_statistic() is not implemented!\n");
			return 0.0;
		}

		virtual float64_t compute_p_value(float64_t statistic)
		{
			SG_ERROR("%s::compute_p_value() is not implemented!\n");
			return 0.0;
		}

		virtual float64_t compute_threshold(float64_t alpha)
		{
			SG_ERROR("%s::compute_threshold() is not implemented!\n");
			return 0.0;
		}

		inline virtual const char* get_name() const=0;
};

}

#endif /* __TESTSTATISTIC_H_ */
