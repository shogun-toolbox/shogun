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

#ifndef INIT_PER_FEATURE_H__
#define INIT_PER_FEATURE_H__

#include <memory>
#include <shogun/lib/common.h>

namespace shogun
{

class CFeatures;

namespace internal
{

class DataFetcher;
class DataManager;

class InitPerFeature
{
	friend class DataManager;
private:
	explicit InitPerFeature(std::unique_ptr<DataFetcher>& fetcher);
public:
	~InitPerFeature();
	InitPerFeature& operator=(CFeatures* feats);
	operator const CFeatures*() const;
private:
	std::unique_ptr<DataFetcher>& m_fetcher;
};

}

}

#endif // INIT_PER_FEATURE_H__

