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
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/lib/config.h>

namespace shogun
{
	class Labels;

	/** @brief Implements a timeseries splitting strategy for cross-validation.
	 * The strategy builds a given number of subsets each filled with labels
	 * indices greater than a split index. The split indices are \f$ c[N/K] \f$
	 * where \f$N\f$ is number of labels,\f$K\f$ is number of subsets and
	 * \f$c = 1,2,3,...,K-1\f$. The last split index is \f$ N-h \f$.
	 *
	 * If forecasting h-step ahead, set minimum subset size to h(Default is 1).
	 *
	 * Given, \f$h = 2\f$ and \f$K = 2\f$ and \f$A = [0,1,2,3,4,5,6,7,8,9]\f$.
	 * The two splits are \f$ S1 = [5,6,7,8,9] \f$ and \f$ S2 = [8,9]\f$
	 */

	class TimeSeriesSplitting : public RandomMixin<SplittingStrategy>
	{
	public:
		/** constructor */
		TimeSeriesSplitting();

		/** constructor
		 *
		 * @param labels labels to be (possibly) used for splitting
		 * @param num_subsets desired number of subsets, the labels are split
		 * into
		 */
		TimeSeriesSplitting(std::shared_ptr<Labels> labels, index_t num_subsets);

		/** Sets the minimum subset size for subsets. If forecasting h-step
		 * ahead, set min_size to h.
		 *
		 * @param min_size Minimum subset size. */
		void set_min_subset_size(index_t min_size);

		/** @return Minimum subset size. */
		index_t get_min_subset_size();

		/** @return name of the SGSerializable */
		virtual const char* get_name() const override
		{
			return "TimeSeriesSplitting";
		}

		void build_subsets() override;

		/**  The minimum subset size for test set.*/
		index_t m_min_subset_size;

	private:
		void init();
	};
}

#endif /* __TIMESERIESSPLITTING_H_ */
