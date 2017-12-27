/*
 * Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2017 Sahil Chaddha
 */

#ifndef __TIMESERIESSPLITTING_H_
#define __TIMESERIESSPLITTING_H_

#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/lib/config.h>

namespace shogun
{
	class CLabels;
	/** @brief Implements a timeseries splitting strategy for cross-validation,
	 * respecting time.
	 * Each fold splits timeseries into train (subset_inverse) and validation
	 * (subset) set.
	 * Train set contains indices less than a split index
	 * and validation set contains rest of the indices.
	 * The split indices are \f$ C*floor(N/K) \f$
	 * where \f$N\f$ is number of labels,\f$K\f$ is number of subsets and
	 * \f$C = 1,2,3,...,K-1\f$. The split index for last fold is \f$ N -
	 * min_future_steps \f$. Note:
	 * If forecasting h-step ahead, set min_future_steps to h (default 1).
	 *
	 * For example, if \f$h = 2\f$ and \f$K = 2\f$ and \f$A =
	 * [0,1,2,3,4,5,6,7,8,9]\f$.
	 * The two folds give the following train and vaildation sets :
	 *
	 * \f$ Train_1 = [0,1,2,3,4], Val_1 = [5,6,7,8,9] \f$ and
	 * \f$ Train_2 = [0,1,2,3,4,5,6,7], Val_2 = [8,9]\f$
	 */

	class CTimeSeriesSplitting : public CSplittingStrategy
	{
	public:
		/** Constructor */
		CTimeSeriesSplitting();

		/** Constructor
		 *
		 * @param labels labels required for the size of timeseries to split
		 * @param num_folds number of folds
		 */
		CTimeSeriesSplitting(CLabels* labels, index_t num_folds);

		/** Sets the minimum number of future steps required in validation set
		 * (default 1).
		 * If forecasting h-step
		 * ahead, set future_steps to h.
		 *
		 * @param future_steps Minimum number of future steps. */
		void set_min_future_steps(index_t future_steps);

		/** @return Minimum number of future steps in validation set. */
		index_t get_min_future_steps();

		/** @return Name of the SGSerializable */
		virtual const char* get_name() const
		{
			return "TimeSeriesSplitting";
		}

		void build_subsets() override;

		/**  The minimum number of future_steps in validation set.*/
		index_t m_min_future_steps;

	private:
		void init();
	};
}

#endif /* __TIMESERIESSPLITTING_H_ */