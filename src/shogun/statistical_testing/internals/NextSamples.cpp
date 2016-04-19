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

#include <shogun/features/Features.h>
#include <shogun/statistical_testing/internals/NextSamples.h>

using namespace shogun;
using namespace internal;

NextSamples::NextSamples(index_t num_distributions) : m_num_blocks(0)
{
	next_samples.resize(num_distributions);
}

NextSamples& NextSamples::operator=(const NextSamples& other)
{
	clear();
	m_num_blocks=other.m_num_blocks;
	next_samples=other.next_samples;
	return *this;
}

NextSamples::~NextSamples()
{
	clear();
}

std::vector<Block>& NextSamples::operator[](size_t i)
{
	REQUIRE(i>=0 && i<next_samples.size(),
			"index (%d) must be between [0,%d]!\n",
			i, next_samples.size()-1);
	return next_samples[i];
}

const std::vector<Block>& NextSamples::operator[](size_t i) const
{
	REQUIRE(i>=0 && i<next_samples.size(),
			"index (%d) must be between [0,%d]!\n",
			i, next_samples.size()-1);
	return next_samples[i];
}

const index_t NextSamples::num_blocks() const
{
	return m_num_blocks;
}

const bool NextSamples::empty() const
{
	using type=const std::vector<Block>;
	return std::any_of(next_samples.cbegin(), next_samples.cend(), [](type& f) { return f.size()==0; });
}

void NextSamples::clear()
{
	using type=std::vector<Block>;
	std::for_each(next_samples.begin(), next_samples.end(), [](type& f) { f.clear(); });
	next_samples.clear();
}
