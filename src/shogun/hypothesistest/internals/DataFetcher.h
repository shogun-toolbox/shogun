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
#include <shogun/hypothsistest/internals/BlockwiseDetails.h>

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
	virtual void start();
	virtual std::shared_ptr<CFeatures> next();
	virtual void reset();
	virtual void end();
	const index_t get_num_samples() const;
	BlockwiseDetails& fetch_blockwise();
	virtual const char* get_name() const;
protected:
	DataFetcher();
	BlockwiseDetails m_block_details;
	index_t m_num_samples;
private:
	std::shared_ptr<CFeatures> m_samples;
};

}

}
#endif // DATA_FETCHER_H__
