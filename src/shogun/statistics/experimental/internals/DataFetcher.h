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

#include <memory>
#include <shogun/lib/common.h>
#include <shogun/statistics/experimental/internals/BlockwiseDetails.h>

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
