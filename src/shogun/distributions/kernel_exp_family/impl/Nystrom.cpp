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

#include "kernel/Base.h"
#include "Nystrom.h"

#include "Full.h"

using namespace shogun;
using namespace shogun::kernel_exp_family_impl;
using namespace Eigen;

Nystrom::Nystrom(SGMatrix<float64_t> data, SGMatrix<float64_t> basis,
		kernel::Base* kernel, float64_t lambda) : Full(basis, kernel, lambda)
{
	SG_SINFO("Using m=%d user provided RKHS basis functions.\n", basis.num_cols);
	m_data = data;
}

Nystrom::Nystrom(SGMatrix<float64_t> data, SGVector<index_t> basis_inds,
			kernel::Base* kernel, float64_t lambda) : Full(data, kernel, lambda)
{
	SG_SINFO("Using m=%d user provided subsample of data RKHS basis functions.\n",
			basis_inds.vlen);
	m_basis = subsample_matrix_cols(basis_inds, m_data);

	SG_SINFO("Avoid double precomputation of kernel.\n");
	m_kernel->set_lhs(data);
	SG_SINFO("Problem size is m=%d, D=%d.\n", get_num_basis(), get_num_dimensions());
	m_kernel->precompute();
}

Nystrom::Nystrom(SGMatrix<float64_t> data, index_t num_subsample_basis,
			kernel::Base* kernel, float64_t lambda) : Full(data, kernel, lambda)
{
	auto N = get_num_data();
	auto D = get_num_dimensions();


	SG_SINFO("Using m*D=%d*%d uniformly sampled RKHS basis functions.\n",
			num_subsample_basis, D);
	auto basis_inds = choose_m_in_n(num_subsample_basis, N);
	m_basis = subsample_matrix_cols(basis_inds, m_data);

	SG_SINFO("Avoid double precomputation of kernel.\n");
	m_kernel->set_lhs(data);
	SG_SINFO("Problem size is m=%d, D=%d.\n", get_num_basis(), get_num_dimensions());
	m_kernel->precompute();
}

void Nystrom::fit()
{
	auto D = get_num_dimensions();
	auto N = get_num_data();
	auto m = get_num_basis();
	auto ND = N*D;
	auto mD = m*D;

	// keep references for later
	auto basis = m_basis;
	auto data = m_data;

	SG_SINFO("Computing h.\n");
	set_basis_and_data(m_data, m_data);
	auto h = compute_h();
	auto eigen_h=Map<VectorXd>(h.vector, ND);

	SG_SINFO("Computing sub-sampled kernel Hessians.\n");
	set_basis_and_data(basis, basis);
	auto G_mm = m_kernel->dx_dy_all();
	Map<MatrixXd> eigen_G_mm(G_mm.matrix, mD, mD);

	SG_SINFO("TODO: Redundant when sub-sampling basis from data.\n");
	set_basis_and_data(basis, data);
	auto G_mn = m_kernel->dx_dy_all();
	Map<MatrixXd> eigen_G_mn(G_mn.matrix, mD, ND);

	eigen_G_mm *= N*m_lambda;
	eigen_G_mm += eigen_G_mn*eigen_G_mn.adjoint();

	m_beta = SGVector<float64_t>(mD);
	auto eigen_beta = Map<VectorXd>(m_beta.vector, mD);

	SG_SINFO("Solving with HouseholderQR.\n");
	auto solver = HouseholderQR<MatrixXd>();
	solver.compute(eigen_G_mm);

//	if (!solver.info() == Eigen::Success)
//		SG_SWARNING("Numerical problems computing HouseholderQR.\n");

	eigen_beta = solver.solve(eigen_G_mn * (eigen_h / m_lambda));
	return;

	SG_SINFO("Computing pseudo-inverse.\n");
	// SG_SWARNING("TODO: compare to CG in terms of speed.\n");
	auto G_dagger = pinv_self_adjoint(G_mm);
	Map<MatrixXd> eigen_G_dagger(G_dagger.matrix, mD, mD);

	SG_SINFO("Constructing solution.\n");
	eigen_beta = eigen_G_dagger * eigen_G_mn * (eigen_h / m_lambda);
}

SGMatrix<float64_t> Nystrom::pinv_self_adjoint(const SGMatrix<float64_t>& A)
{
	// based on the snippet from
	// http://eigen.tuxfamily.org/index.php?title=FAQ#Is_there_a_method_to_compute_the_.28Moore-Penrose.29_pseudo_inverse_.3F
	// modified using eigensolver for psd problems
	auto m=A.num_rows;
	ASSERT(A.num_cols == m);
	auto eigen_A=Map<MatrixXd>(A.matrix, m, m);

	SelfAdjointEigenSolver<MatrixXd> solver(eigen_A);
	auto s = solver.eigenvalues();
	auto V = solver.eigenvectors();

	// tol = eps⋅max(m,n) * max(singularvalues)
	// this is done in numpy/Octave & co
	// c.f. https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse#Singular_value_decomposition_.28SVD.29
	float64_t pinv_tol = CMath::MACHINE_EPSILON * m * s.maxCoeff();

	VectorXd inv_s(m);
	for (auto i=0; i<m; i++)
	{
		if (s(i) > pinv_tol)
			inv_s(i)=1.0/s(i);
		else
			inv_s(i)=0;
	}

	SGMatrix<float64_t> A_pinv(m, m);
	Map<MatrixXd> eigen_pinv(A_pinv.matrix, m, m);
	eigen_pinv = (V*inv_s.asDiagonal()*V.transpose());

	return A_pinv;
}

SGVector<index_t> Nystrom::choose_m_in_n(index_t m, index_t n, bool sorted)
{
	SGVector<index_t> permutation(n);
	permutation.range_fill();
	CMath::permute(permutation);
	auto chosen = SGVector<index_t>(m);

	memcpy(chosen.vector, permutation.vector, sizeof(index_t)*m);

	// in order to have more sequential data reads
	if (sorted)
		CMath::qsort(chosen.vector, m);

	return chosen;
}

SGMatrix<float64_t> Nystrom::subsample_matrix_cols(SGVector<index_t> col_inds,
		SGMatrix<float64_t> mat)
{
	auto subsampled=SGMatrix<float64_t>(mat.num_rows, col_inds.vlen);
	for (auto i=0; i<col_inds.vlen; i++)
	{
		memcpy(subsampled.get_column_vector(i), mat.get_column_vector(col_inds[i]),
				sizeof(float64_t)*mat.num_rows);
	}

	return subsampled;
}
