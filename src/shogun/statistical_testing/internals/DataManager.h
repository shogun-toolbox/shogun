/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 - 2016 Soumyajit De
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

#ifndef DATA_MANAGER_H__
#define DATA_MANAGER_H__

#include <vector>
#include <memory>
#include <shogun/statistical_testing/internals/InitPerFeature.h>
#include <shogun/lib/common.h>

namespace shogun
{

class CFeatures;

namespace internal
{

class DataFetcher;
class NextSamples;

/**
 * @brief Class DataManager for fetching/streaming test data block-wise.
 * It can handle data coming from multiple sources. The number of data
 * sources is represented by the num_distributions parameter in the constructor
 * of the data manager. It can handle heterogenous data sources, and it can
 * stream multiple blocks per burst, as the computation would require. The size
 * of the blocks and the number of blocks to be fetched per burst can be set
 * externally.
 *
 * This class is designed to be used on a stack. An instance of DataManager
 * should not be serialzied or copied or moved around. In Shogun, it is helpful
 * when used inside just the implementation inside a PIMPL.
 */
class DataManager
{
public:
	/**
	 * Default constructor.
	 *
	 * @param num_distributions number of data sources (i.e. CFeature objects)
	 */
	DataManager(size_t num_distributions);

	/**
	 * Disabled copy constructor
	 * @param other other instance
	 */
	DataManager(const DataManager& other) = delete;

	/**
	 * Disabled assignment operator
	 * @param other other instance
	 */
	DataManager& operator=(const DataManager& other) = delete;

	/**
	 * Destructor
	 */
	~DataManager();

	/**
	 * Sets the blocksize for block-wise data fetching. It divides the block-size
	 * per data source according to the total number of feature vectors available
	 * from that source. More formally, if there are \f$K\f$ data sources, \f$X_k\f$,
	 * \f$k=\[1,K]\f$, with number of feature vectors \f$n_{X_k}\f$ from each, then
	 * setting a block-size of \f$B\f$ would mean that in each next() call of the
	 * data manager instance, it will fetch \f$rho_{X_k} B\f$ samples from each
	 * \f$X_k\f$, where \f$rho_{X_k}=n_{X_k}/n\f$, \f$n=sum_k n_{X_k}\f$.
	 *
	 * @param blocksize The size of the block consisting of data from all the sources.
	 */
	void set_blocksize(index_t blocksize);

	/**
	 * In order to speed up the computation, usually a number of blocks are fetched at
	 * once per next() call. This method sets that number.
	 *
	 * @param num_blocks_per_burst The number of blocks to be fetched in a burst.
	 */
	void set_num_blocks_per_burst(index_t num_blocks_per_burst);

	/**
	 * Setter for feature object as a data source. Since multiple data sources are
	 * supported, this method takes an index in which the feature object is set.
	 * Internally, it initializes a data fetcher object for the provided feature
	 * object.
	 *
	 * Example usage:
	 * @code
	 *
	 * DataManager data_mgr;
	 * // feats_0 = some CFeatures instance
	 * // feats_1 = some CFeatures instance
	 * data_mgr.sample_at(0) = feats_0;
	 * data_mgr.sample_at(1) = feats_1;
	 *
	 * @endcode
	 *
	 * @param i The data source index, at which the feature object is to be set as a
	 * data source.
	 * @return An initializer for the specified data source (that sets up a fetcher
	 * for this feature), to be used as lvalue.
	 */
	InitPerFeature samples_at(size_t i);

	/**
	 * Getter for feature object at a give data source index.
	 *
	 * @param i The data source index, from which the feature object is to be obtained
	 * @return The underlying CFeatures object at the specified data source.
	 */
	CFeatures* samples_at(size_t i) const;

	/**
	 * Setter for the number of samples. Setting this number is mandatory for
	 * streaming features. For other type of feature objects, this number equals
	 * the number of vectors, and is set internally.
	 *
	 * Example usage:
	 * @code
	 *
	 * DataManager data_mgr;
	 * data_mgr.num_sample_at(0) = 10;
	 * data_mgr.num_sample_at(1) = 15;
	 *
	 * @endcode
	 *
	 * @param i The data source index, at which the number of samples is to be set.
	 * @return A reference for the number of samples for the specified data source
	 * to be used as lvalue.
	 */
	index_t& num_samples_at(size_t i);

	/**
	 * Getter for the number of samples.
	 *
	 * @param i The data source index, from which the number of samples is to be obtained.
	 * @return The number of samples for the specified data source.
	 */
	const index_t num_samples_at(size_t i) const;

	/**
	 * Getter for the number of samples from a specified data source in a block.
	 *
	 * @param i The data source index.
	 * @return The number of samples from i-th data source in a block.
	 */
	const index_t blocksize_at(size_t i) const;

	/**
	 * @return Total number of samples that can be fetched from all the data sources.
	 */
	index_t get_num_samples() const;

	/**
	 * @return The minimum block-size that can be fetched from the specified data sources.
	 * For example, if there are two data sources, with samples 20 and 30, respectively,
	 * then minimum blocksize can be 5 (2 from 1st data source, 3 from the 2nd), and there
	 * can be then 10 such blocks.
	 */
	index_t get_min_blocksize() const;

	/**
	 * Call this method before fetching the data from the data manager
	 */
	void start();

	/**
	 * @return The next bunch of blocks fetched at any given burst.
	 */
	NextSamples next();

	/**
	 * call this method after fetching the data is done.
	 */
	void end();

	/**
	 * Resets the fetchers to the initial states.
	 */
	void reset();
private:
	/**
	 * The internal data fetcher instances.
	 */
	std::vector<std::unique_ptr<DataFetcher>> fetchers;
};

}

}

#endif // DATA_MANAGER_H__
