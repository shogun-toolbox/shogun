/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

#include "Base.h"

using namespace shogun;
using namespace shogun::kernel_exp_family_impl::kernel;
using namespace Eigen;

index_t Base::get_num_dimensions() const
{
	return m_lhs.num_rows;
}

index_t Base::get_num_lhs() const
{
	return m_lhs.num_cols;
}

void Base::set_rhs(SGMatrix<float64_t> rhs)
{
	m_rhs = rhs;
}

void Base::set_rhs(SGVector<float64_t> rhs)
{
	set_rhs(SGMatrix<float64_t>(rhs));
}

void Base::set_lhs(SGMatrix<float64_t> lhs)
{
	m_lhs = lhs;
}

void Base::set_lhs(SGVector<float64_t> lhs)
{
	set_lhs(SGMatrix<float64_t>(lhs));
}

index_t Base::get_num_rhs() const
{
	return m_rhs.num_cols;
}

SGMatrix<float64_t> Base::kernel_all() const
{
    auto N_lhs = get_num_lhs();
    auto N_rhs = get_num_rhs();

    SGMatrix<float64_t> result(N_lhs, N_rhs);

#pragma omp parallel for
    for (auto idx_a = 0; idx_a < N_lhs; ++idx_a) {
        for (auto idx_b = 0; idx_b < N_rhs; ++idx_b) {
            result.set_element(this->kernel(idx_a, idx_b), idx_a, idx_b);
        }
    }

    return result;
}

SGMatrix<float64_t> Base::dx_all() const
{
    auto D = get_num_dimensions();
    auto N_lhs = get_num_lhs();
    auto N_rhs = get_num_rhs();
    auto ND_lhs = N_lhs*D;

    SGMatrix<float64_t> result(ND_lhs, N_rhs);
    Map<MatrixXd> eigen_result(result.matrix, ND_lhs, N_rhs);

    // TODO exploit symmetry computation
    // TODO exploit symmetry in storage (Shognu lib?)
    // TODO the assignment in matrix is not sequentially in memory, does it matter?
#pragma omp parallel for
    for (auto idx_a=0; idx_a<N_lhs; idx_a++)
        for (auto idx_b=0; idx_b<N_rhs; idx_b++)
        {
            auto row_start = idx_a*D;
            auto col_start = idx_b;
            SGVector<float64_t> h = dx(idx_a, idx_b);
            eigen_result.block(row_start, col_start, D, 1) = Map<MatrixXd>(h.vector, D, 1);
        }

    return result;
}

SGMatrix<float64_t> Base::dy_all() const
{
    auto D = get_num_dimensions();
    auto N_lhs = get_num_lhs();
    auto N_rhs = get_num_rhs();
    auto ND_rhs = N_rhs*D;

    SGMatrix<float64_t> result(N_lhs, ND_rhs);
    Map<MatrixXd> eigen_result(result.matrix, N_lhs, ND_rhs);

    // TODO exploit symmetry computation
    // TODO exploit symmetry in storage (Shognu lib?)
    // TODO the assignment in matrix is not sequentially in memory, does it matter?
#pragma omp parallel for
    for (auto idx_a=0; idx_a<N_lhs; idx_a++)
        for (auto idx_b=0; idx_b<N_rhs; idx_b++)
        {
            auto row_start = idx_a;
            auto col_start = idx_b*D;
            SGVector<float64_t> h = dy(idx_a, idx_b);
            eigen_result.block(row_start, col_start, 1, D) = Map<MatrixXd>(h.vector, 1, D);
        }

    return result;
}

SGMatrix<float64_t> Base::dx_dy_all() const
{
    auto D = get_num_dimensions();
    auto N_lhs = get_num_lhs();
    auto N_rhs = get_num_rhs();
    auto ND_lhs = N_lhs*D;
    auto ND_rhs = N_rhs*D;

    SGMatrix<float64_t> result(ND_lhs,ND_rhs);
    Map<MatrixXd> eigen_result(result.matrix, ND_lhs,ND_rhs);

    // TODO exploit symmetry computation
    // TODO exploit symmetry in storage (Shognu lib?)
    // TODO the assignment in matrix is not sequentially in memory, does it matter?
#pragma omp parallel for
    for (auto idx_a=0; idx_a<N_lhs; idx_a++)
        for (auto idx_b=0; idx_b<N_rhs; idx_b++)
        {
            auto row_start = idx_a*D;
            auto col_start = idx_b*D;
            SGMatrix<float64_t> h=dx_dy(idx_a, idx_b);
            eigen_result.block(row_start, col_start, D, D) = Map<MatrixXd>(h.matrix, D, D);
        }

    return result;
}


Base::Base()
{
}

