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

/** enum for different method to compute p-value of test. To estimate p-value
 * the null distribution somehow needs to be approximated. This enum defines
 * the used method */
enum EPValueMethod
{
	BOOTSTRAP, MMD2_SPECTRUM, MMD2_GAMMA
};

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

		/** sets the method how to approximate the null-distribution
		 * @param p-value method to use
		 */
		virtual void set_p_value_method(EPValueMethod p_value_method);

		/** computes a p-value based on bootstrapping the null-distribution.
		 * This method should be overridden for different methods
		 *
		 * @param statistic statistic value to compute the p-value for
		 * @return p-value parameter statistic is the (1-p) percentile of the
		 * null distribution
		 */
		virtual float64_t compute_p_value(float64_t statistic);

		/** Performs a two-sample test with the current settings on the current
		 * data. Computes test statistic and compares its p-value against the
		 * desired one and returns true if the p-value is at least as good.
		 *
		 * @param alpha test niveau alpha
		 * @return true if null-hypothesis (p==q) is rejected, false otherwise
		 */
		virtual bool perform_test(float64_t alpha);

		virtual ~CTwoSampleTestStatistic();

		inline virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		/** concatenated samples of the two distributions (two blocks) */
		CFeatures* m_p_and_q;

		/** defines the first index of samples of q */
		index_t m_q_start;

		/** number of iterations for bootstrapping null-distributions */
		index_t m_bootstrap_iterations;

		/** Defines how the p-value for the null distribution is computed */
		EPValueMethod m_p_value_method;
};

}

#endif /* __TWOSAMPLETESTSTATISTIC_H_ */
