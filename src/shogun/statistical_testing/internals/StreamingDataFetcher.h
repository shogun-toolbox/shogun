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

#include <shogun/lib/common.h>
#include <shogun/statistical_testing/internals/DataFetcher.h>

#ifndef STREMING_DATA_FETCHER_H__
#define STREMING_DATA_FETCHER_H__

namespace shogun
{

class StreamingFeatures;

namespace internal
{

class DataManager;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
class StreamingDataFetcher : public DataFetcher
{
	friend class DataManager;
public:
	StreamingDataFetcher(const std::shared_ptr<StreamingFeatures>& samples);
	~StreamingDataFetcher() override;
	void set_num_samples(index_t num_samples);

	void shuffle_features() override;
	void unshuffle_features() override;

	void use_fold(index_t i) override;
	void init_active_subset() override;

	void start() override;
	std::shared_ptr<Features> next() override;
	void reset() override;
	void end() override;

	index_t get_num_samples() const override;
	const char* get_name() const override
	{
		return "StreamingDataFetcher";
	}
private:
	std::shared_ptr<StreamingFeatures> m_samples;
	bool parser_running;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS
}

}
#endif // STREMING_DATA_FETCHER_H__
