/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NEXT_SAMPLES_H__
#define NEXT_SAMPLES_H__

#include <vector>
#include <memory>
#include <shogun/lib/common.h>

namespace shogun
{

class CFeatures;

namespace internal
{

/**
 * @brief class NextSamples is the return type for next() call in DataManager.
 * If there are no more samples (from any one of the distributions), an empty
 * instance of NextSamples is supposed to be returned. This can be verified
 * from the caller by calling the empty() method. Otherwise, always a get()
 * call with appropriate index would give the samples from that distribution.
 * If an inappropriate index is provided, e.g. get(2) for a two-sample test,
 * a runtime exception is thrown.
 *
 * Example usage:
 * @code
 * 		NextSamples next_samples(2);
 * 		next_samples[0] = fetchers[0].next();
 * 		next_samples[1] = fetchers[1].next();
 * 		if (!next_samples.empty())
 * 		{
 * 			auto first = next_samples[0];
 * 			auto second = next_samples[1];
 * 			auto third = next_samples[2]; // Runtime Error
 * 		}
 * @endcode
 */
class NextSamples
{
	friend class DataManager;
private:
	NextSamples(index_t num_distributions);
public:
	~NextSamples();
	/**
	 * Contains a number of blocks (of samples) fetched in the current burst from a
	 * specified distribution.
	 *
	 * @param i determines samples from which distribution
	 * @return a vector of fetched blocks of features from the specified distribution
	 */
	std::vector<std::shared_ptr<CFeatures>>& operator[](index_t i);

	/**
	 * Const version of the above. This is called when a const instance of NextSamples
	 * is returned.
	 */
	const std::vector<std::shared_ptr<CFeatures>>& operator[](index_t i) const;

	/**
	 * @return number of blocks fetched from each of the distribution. It is assumed
	 * that this number is same for all the distributions.
	 */
	const index_t num_blocks() const;

	/**
	 * This returns true if any of the distribution fetched 0 blocks (checked from the
	 * size of the vector for that distribution)
	 *
	 * @return whether this instance does not contain any blocks of samples from any
	 * of the distribution
	 */
	const bool empty() const;
private:
	index_t m_num_blocks;
	std::vector<std::vector<std::shared_ptr<CFeatures>>> next_samples;
};

}

}

#endif // NEXT_SAMPLES_H__
