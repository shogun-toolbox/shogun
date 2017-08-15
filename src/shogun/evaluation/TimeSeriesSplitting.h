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

	/** @brief Implementation of timeseries splitting for cross-validation.
	 * Produces subsets each filled with indices greater than a split point.
	 */

	class CTimeSeriesSplitting : public CSplittingStrategy
	{
	public:
		/** constructor */
		CTimeSeriesSplitting();

		/** constructor
		 *
		 * @param labels labels to be (possibly) used for splitting
		 * @param num_subsets desired number of subsets, the labels are split
		 * into
		 */
		CTimeSeriesSplitting(CLabels* labels, index_t num_subsets);

		/** @param h future needed in test set. */
		void set_h(index_t h);

		/** @return h value. */
		index_t get_h();

		/** @return name of the SGSerializable */
		virtual const char* get_name() const
		{
			return "TimeSeriesSplitting";
		}

		/** implementation of the time-series cross-validation splitting
		 * strategy */
		virtual void build_subsets();

		/** custom rng if using cross validation across different threads */
		CRandom* m_rng;

		/** atleast h size of each subsets.(default 1) */
		index_t m_h = 1;
	};
}

#endif /* __TIMESERIESSPLITTING_H_ */