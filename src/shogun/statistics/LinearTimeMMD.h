/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __LINEARTIMEMMD_H_
#define __LINEARTIMEMMD_H_

#include <shogun/statistics/TwoSampleTestStatistic.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;

/** enum for different method to compute p-value of test, MMD_NONE will result
 * in calling CTwoSampleTestStatistic::compute_p_value, where bootstrapping
 * is implemented */
enum EMMDThreshold
{
	MMD_NONE
};

class CLinearTimeMMD : public CTwoSampleTestStatistic
{
	public:
		CLinearTimeMMD();
		CLinearTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t q_start);

		virtual ~CLinearTimeMMD();

		virtual float64_t compute_statistic();
		virtual float64_t compute_p_value(float64_t statistic);

		inline virtual const char* get_name() const
		{
			return "LinearTimeMMD";
		};

	private:
		void init();

	protected:
		CKernel* m_kernel;

		EMMDThreshold m_threshold_method;
		index_t m_bootstrap_iterations;

};

}

#endif /* __LINEARTIMEMMD_H_ */
