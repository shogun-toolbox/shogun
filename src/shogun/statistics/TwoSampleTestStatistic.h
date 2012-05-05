/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __TWOSAMPLETESTSTATISTIC_H_
#define __TWOSAMPLETESTSTATISTIC_H_

#include <shogun/statistics/TestStatistic.h>

namespace shogun
{

class CFeatures;

class CTwoSampleTestStatistic : public CTestStatistic
{
	public:
		CTwoSampleTestStatistic();
		CTwoSampleTestStatistic(CFeatures* p_and_q, index_t q_start);

		virtual ~CTwoSampleTestStatistic();

		inline virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		CFeatures* m_p_and_q;
		index_t m_q_start;
};

}

#endif /* __TWOSAMPLETESTSTATISTIC_H_ */
