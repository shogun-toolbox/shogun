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

#include <algorithm>
#include <iostream>
#include <shogun/hypothsistest/internals/NextSamples.h>
#include <shogun/features/Features.h>

using namespace shogun;
using namespace internal;

NextSamples::NextSamples(index_t num_distributions) : m_num_blocks(0)
{
	next_samples.resize(num_distributions);
}

NextSamples::~NextSamples()
{
}

std::vector<std::shared_ptr<CFeatures>>& NextSamples::operator[](index_t i)
{
//	std::cout << "NextSamples::acessing fetched sample at " << i << " using non-const access operator" << std::endl;
	REQUIRE(i >= 0 && i < next_samples.size(), "index (%d) must be between [0,%d]!\n", i, next_samples.size() - 1);
	return next_samples[i];
}

const std::vector<std::shared_ptr<CFeatures>>& NextSamples::operator[](index_t i) const
{
//	std::cout << "NextSamples::acessing fetched sample at " << i << " using const access operator" << std::endl;
	REQUIRE(i >= 0 && i < next_samples.size(), "index (%d) must be between [0,%d]!\n", i, next_samples.size() - 1);
	return next_samples[i];
}

const index_t NextSamples::num_blocks() const
{
	return m_num_blocks;
}

const bool NextSamples::empty() const
{
	using type = const std::vector<std::shared_ptr<CFeatures>>;
	return std::any_of(next_samples.cbegin(), next_samples.cend(), [](type& f) { return f.size() == 0; });
}
