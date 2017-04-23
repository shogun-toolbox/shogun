/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2016 - 2017 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
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
#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif // DOXYGEN_SHOULD_SKIP_THIS
}

}

}

#endif // WITHIN_BLOCK_PERMUTATION_H_
