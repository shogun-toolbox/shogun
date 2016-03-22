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
 * along with this program.  If not, see <http:/www.gnu.org/licenses/>.
 */

#ifndef DATA_MANAGER_H__
#define DATA_MANAGER_H__

#include <vector>
#include <memory>
#include <shogun/hypothesistest/internals/InitPerFeature.h>
#include <shogun/lib/common.h>

namespace shogun
{

class CFeatures;

namespace internal
{

class DataFetcher;
class NextSamples;

class DataManager
{
public:
	DataManager(index_t num_distributions);
	DataManager(const DataManager& other) = delete;
	DataManager& operator=(const DataManager& other) = delete;
	~DataManager();

	void set_blocksize(index_t blocksize);
	void set_num_blocks_per_burst(index_t num_blocks_per_burst);

	InitPerFeature samples_at(index_t i);
	CFeatures* samples_at(index_t i) const;

	index_t& num_samples_at(index_t i);
	const index_t num_samples_at(index_t i) const;

	const index_t blocksize_at(index_t i) const;

	index_t get_num_samples() const;
	index_t get_min_blocksize() const;

	void start();
	NextSamples next();
	void end();
	void reset();
private:
	std::vector<std::unique_ptr<DataFetcher>> fetchers;
};

}

}

#endif // DATA_MANAGER_H__
