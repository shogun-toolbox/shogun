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

#ifndef TEST_TYPES_H__
#define TEST_TYPES_H__

#include <vector>
#include <shogun/lib/config.h>

namespace shogun
{

class CFeatures;

namespace internal
{

struct TwoSampleTestPermutationPolicy;
struct IndependenceTestPermutationPolicy;

struct OneDistributionTest
{
	enum { num_feats = 1 };
};

struct TwoDistributionTest
{
	enum { num_feats = 2 };
};

struct ThreeDistributionTest
{
	enum { num_feats = 3 };
};

struct GoodnessOfFitTest : OneDistributionTest
{
	enum { num_kernels = 1 };
	using return_type = std::shared_ptr<CFeatures>;
};

struct TwoSampleTest : TwoDistributionTest
{
	enum { num_kernels = 1 };
	using permutation_policy = TwoSampleTestPermutationPolicy;
	using return_type = std::shared_ptr<CFeatures>;
};

struct IndependenceTest : TwoDistributionTest
{
	enum { num_kernels = 2 };
	using permutation_policy = IndependenceTestPermutationPolicy;
	using return_type = std::vector<std::shared_ptr<CFeatures>>;
};

}

}

#endif // TEST_TYPES_H__
