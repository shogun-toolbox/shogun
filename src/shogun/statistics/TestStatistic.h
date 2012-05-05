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

		virtual float64_t compute_threshold(float64_t confidence)
		{
			SG_ERROR("%s::compute_threshold() is not implemented!\n");
			return 0.0;
		}

		inline virtual const char* get_name() const=0;
};

}

#endif /* __TESTSTATISTIC_H_ */
