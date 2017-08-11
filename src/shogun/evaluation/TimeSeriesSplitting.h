/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
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