/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2014  Soumyajit De
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

#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockPermutation.h>
#include <shogun/statistical_testing/internals/mmd/BiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedIncomplete.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

WithinBlockPermutation::WithinBlockPermutation(index_t n, EStatisticType type)
: n_x(n), stype(type)
{
}

float64_t WithinBlockPermutation::operator()(SGMatrix<float64_t> km)
{
	SG_SDEBUG("Entering!\n");
	inds.resize(km.num_rows);
	std::iota(inds.data(), inds.data()+inds.size(), 0);
	SGVector<index_t> permuted_inds(inds.data(), inds.size(), false);
	CMath::permute(permuted_inds);

	const index_t n_y=km.num_rows-n_x;
	SG_SDEBUG("number of samples are %d and %d!\n", n_x, n_y);

	auto term_1=0.0;
	for (auto i=0; i<n_x; ++i)
	{
		for (auto j=0; j<n_x; ++j)
		{
			if (i>j)
				term_1+=km(permuted_inds[i], permuted_inds[j]);
		}
	}
	term_1*=2;
	SG_SDEBUG("term_1 sum (without diagonal) = %f!\n", term_1);
	if (stype==EStatisticType::BIASED_FULL)
	{
		for (auto i=0; i<n_x; ++i)
			term_1+=km(permuted_inds[i], permuted_inds[i]);
		SG_SDEBUG("term_1 sum (with diagonal) = %f!\n", term_1);
		term_1/=n_x*n_x;
	}
	else
		term_1/=n_x*(n_x-1);
	SG_SDEBUG("term_1 (normalized) = %f!\n", term_1);

	auto term_2=0.0;
	for (auto i=n_x; i<n_x+n_y; ++i)
	{
		for (auto j=n_x; j<n_x+n_y; ++j)
		{
			if (i>j)
				term_2+=km(permuted_inds[i], permuted_inds[j]);
		}
	}
	term_2*=2.0;
	SG_SDEBUG("term_2 sum (without diagonal) = %f!\n", term_2);
	if (stype==EStatisticType::BIASED_FULL)
	{
		for (auto i=n_x; i<n_x+n_y; ++i)
			term_2+=km(permuted_inds[i], permuted_inds[i]);
		SG_SDEBUG("term_2 sum (with diagonal) = %f!\n", term_2);
		term_2/=n_y*n_y;
	}
	else
		term_2/=n_y*(n_y-1);
	SG_SDEBUG("term_2 (normalized) = %f!\n", term_2);

	auto term_3=0.0;
	for (auto i=n_x; i<n_x+n_y; ++i)
	{
		for (auto j=0; j<n_x; ++j)
			term_3+=km(permuted_inds[i], permuted_inds[j]);
	}
	SG_SDEBUG("term_3 sum (with diagonal) = %f!\n", term_3);
	if (stype==EStatisticType::UNBIASED_INCOMPLETE)
	{
		for (auto i=0; i<n_x; ++i)
			term_3-=km(permuted_inds[i+n_x], permuted_inds[i]);
		SG_SDEBUG("term_3 sum (without diagonal) = %f!\n", term_3);
		term_3/=n_x*(n_x-1);
	}
	else
		term_3/=n_x*n_y;
	SG_SDEBUG("term_3 (normalized) = %f!\n", term_3);

	SG_SDEBUG("Leaving!\n");
	return term_1+term_2-2*term_3;
}
