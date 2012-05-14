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

		/** merges both sets of samples and computes the test statistic
		 * m_bootstrap_iteration times
		 *
		 * @return vector of all statistics
		 */
		virtual SGVector<float64_t> bootstrap_null();

		/** sets the number of bootstrap iterations for bootstrap_null()
		 *
		 * @param bootstrap_iterations how often bootstrapping shall be done
		 */
		void set_bootstrap_iterations(index_t bootstrap_iterations);

		/** computes a p-value based on bootstrapping the null-distribution.
		 * This method should be overridden for different methods
		 *
		 * @param statistic statistic value to compute the p-value for
		 * @return p-value parameter statistic is the (1-p) percentile of the
		 * null distribution
		 */
		virtual float64_t compute_p_value(float64_t statistic);

		virtual ~CTwoSampleTestStatistic();

		inline virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		CFeatures* m_p_and_q;
		index_t m_q_start;

		/** number of iterations for bootstrapping null-distributions */
		index_t m_bootstrap_iterations;
};

}

#endif /* __TWOSAMPLETESTSTATISTIC_H_ */
