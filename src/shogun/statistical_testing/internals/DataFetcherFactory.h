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

#include <memory>
#include <shogun/lib/config.h>

#ifndef DATA_FETCHER_FACTORY_H__
#define DATA_FETCHER_FACTORY_H__

namespace shogun
{

class CFeatures;

namespace internal
{

class DataFetcher;

struct DataFetcherFactory
{
	DataFetcherFactory() = delete;
	DataFetcherFactory(const DataFetcherFactory& other) = delete;
	DataFetcherFactory& operator=(const DataFetcherFactory& other) = delete;
	~DataFetcherFactory() = delete;

	static DataFetcher* get_instance(CFeatures* feats);
};

}

}
#endif // DATA_FETCHER_FACTORY_H__
