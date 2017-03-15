/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 - 2017 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef NEXT_SAMPLES_H__
#define NEXT_SAMPLES_H__

#include <vector>
#include <shogun/lib/common.h>
#include <shogun/statistical_testing/internals/Block.h>

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
 * 			auto third = next_samples[2]; / Runtime Error
 * 		}
 * @endcode
 */
class NextSamples
{
	friend class DataManager;
private:
	NextSamples(index_t num_distributions);
public:
	/**
	 * Assignment operator. Clears the current blocks.
	 */
	NextSamples& operator=(const NextSamples& other);

	/**
	 * Destructor
	 */
	~NextSamples();

	/**
	 * Contains a number of blocks (of samples) fetched in the current burst from a
	 * specified distribution.
	 *
	 * @param i determines samples from which distribution
	 * @return a vector of fetched blocks of features from the specified distribution
	 */
	std::vector<Block>& operator[](size_t i);

	/**
	 * Const version of the above. This is called when a const instance of NextSamples
	 * is returned.
	 */
	const std::vector<Block>& operator[](size_t i) const;

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

	/**
	 * Method that clears the memory occupied by the feature objects inside.
	 */
	void clear();
private:
	index_t m_num_blocks;
	std::vector<std::vector<Block> > next_samples;
};

}

}

#endif // NEXT_SAMPLES_H__
