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

namespace shogun
{

template <typename T> class SGMatrix;
template <typename T> class CGPUMatrix;
enum class EStatisticType;

namespace internal
{

namespace mmd
{

class WithinBlockPermutation
{
	using return_type = float64_t;
public:
	WithinBlockPermutation(index_t, index_t, EStatisticType);
	return_type operator()(SGMatrix<float64_t> kernel_matrix);
//	return_type operator()(CGPUMatrix<float64_t> kernel_matrix);
private:
	const index_t n_x;
	const index_t n_y;
	const EStatisticType stype;
	void add_term(float64_t, index_t, index_t);
	struct terms_t
	{
		float64_t term[3];
		float64_t diag[3];
	};
	terms_t terms;
};

}

}

}

#endif // WITHIN_BLOCK_PERMUTATION_H_
