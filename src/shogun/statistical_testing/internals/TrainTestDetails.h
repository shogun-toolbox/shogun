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

#include <shogun/lib/common.h>

#ifndef TRAIN_TEST_DETAILS_H__
#define TRAIN_TEST_DETAILS_H__

namespace shogun
{

namespace internal
{

/**
 * @brief Class that holds train-test details for the data-fetchers.
 * There are one instance of this class per fetcher.
 */
class TrainTestDetails
{
	friend class DataFetcher;
	friend class StreamingDataFetcher;

public:
	TrainTestDetails();

	void set_total_num_samples(index_t total_num_sampels);
	index_t get_total_num_samples() const;

	void set_num_training_samples(index_t num_training_samples);
	index_t get_num_training_samples() const;
	index_t get_num_test_samples() const;

//	bool is_training_mode() const;
//	void set_train_mode(bool train_mode);
//	void set_xvalidation_mode(bool xvalidation_mode);
private:
	index_t m_total_num_samples;
	index_t m_num_training_samples;
};

}

}
#endif // TRAIN_TEST_DETAILS_H__
