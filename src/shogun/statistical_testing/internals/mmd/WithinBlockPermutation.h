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

#ifndef WITHIN_BLOCK_PERMUTATION_H_
#define WITHIN_BLOCK_PERMUTATION_H_

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/statistical_testing/TestEnums.h>

namespace shogun
{

template <typename T> class SGMatrix;
template <typename T> class CGPUMatrix;

namespace internal
{

namespace mmd
{

class WithinBlockPermutation
{
	typedef float32_t return_type;
public:
	WithinBlockPermutation(index_t, index_t, EStatisticType);
	return_type operator()(const SGMatrix<return_type>& kernel_matrix);
//	return_type operator()(const CGPUMatrix<return_type>& kernel_matrix);
private:
	void add_term(float32_t, index_t, index_t);

	const index_t n_x;
	const index_t n_y;
	const EStatisticType stype;
	SGVector<index_t> permuted_inds;
	SGVector<index_t> inverted_permuted_inds;
	struct terms_t
	{
		float32_t term[3];
		float32_t diag[3];
	};
	terms_t terms;
};

}

}

}

#endif // WITHIN_BLOCK_PERMUTATION_H_
