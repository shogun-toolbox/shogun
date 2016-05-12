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

#ifndef WITHIN_BLOCK_PERMUTATION_BATCH_H_
#define WITHIN_BLOCK_PERMUTATION_BATCH_H_

#include <vector>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

template <typename T> class SGMatrix;
template <typename T> class CGPUMatrix;
enum EStatisticType;

namespace internal
{

namespace mmd
{

class WithinBlockPermutationBatch
{
	using value_type=float32_t;
	using return_type=SGVector<value_type>;
public:
	WithinBlockPermutationBatch(index_t, index_t, index_t, EStatisticType);
	return_type operator()(const SGMatrix<value_type>& kernel_matrix);
//	return_type operator()(const CGPUMatrix<value_type>& kernel_matrix);
private:
	struct terms_t;
	void add_term(terms_t&, float32_t, index_t, index_t);

	const index_t n_x;
	const index_t n_y;
	const index_t num_null_samples;
	const EStatisticType stype;
	SGVector<index_t> permuted_inds;
	std::vector<std::vector<index_t>> inverted_permuted_inds;
};

}

}

}

#endif // WITHIN_BLOCK_PERMUTATION_BATCH_H_
