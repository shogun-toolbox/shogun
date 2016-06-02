/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
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

#include <memory>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/statistical_testing/internals/BlockwiseDetails.h>

#ifndef DATA_FETCHER_H__
#define DATA_FETCHER_H__

namespace shogun
{

class CFeatures;

namespace internal
{

class DataManager;

class DataFetcher
{
	friend class DataManager;
	friend class InitPerFeature;
public:
	DataFetcher(CFeatures* samples);
	virtual ~DataFetcher();

	void set_blockwise(bool blockwise);

	void set_train_test_mode(bool on);
	bool is_train_test_mode() const;

	void set_train_mode(bool on);
	bool is_train_mode() const;

	void set_train_test_ratio(float64_t ratio);
	float64_t get_train_test_ratio() const;

	virtual void shuffle_features();
	virtual void unshuffle_features();

	virtual void use_fold(index_t i);
	virtual void init_active_subset();

	virtual void start();
	virtual CFeatures* next();
	virtual void reset();
	virtual void end();

	virtual index_t get_num_samples() const;

	index_t get_num_folds() const;
	index_t get_num_training_samples() const;
	index_t get_num_testing_samples() const;

	BlockwiseDetails& fetch_blockwise();
	virtual const char* get_name() const
	{
		return "DataFetcher";
	}
protected:
	DataFetcher();
	BlockwiseDetails m_block_details;
	index_t m_num_samples;
	bool train_test_mode;
	bool train_mode;
	float64_t train_test_ratio;
private:
	CFeatures* m_samples;
	SGVector<index_t> shuffle_subset;
	SGVector<index_t> active_subset;
	bool features_shuffled;
	BlockwiseDetails last_blockwise_details;
	void allocate_active_subset();
};

}

}
#endif // DATA_FETCHER_H__
