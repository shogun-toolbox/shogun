/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * Written (w) 2012-2013 Heiko Strathmann
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

#include <shogun/statistics/NOCCO.h>

#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;
using namespace Eigen;

CNOCCO::CNOCCO() : CKernelIndependenceTest()
{
	init();
}

CNOCCO::CNOCCO(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p, CFeatures* q)
	: CKernelIndependenceTest(kernel_p, kernel_q, p, q)
{
	init();

	// only equal number of samples are allowed
	if (p && q)
	{
		REQUIRE(p->get_num_vectors()==q->get_num_vectors(),
				"Only equal number of samples from both the distributions are "
				"possible. Provided %d samples from p and %d samples from q!\n",
				p->get_num_vectors(), q->get_num_vectors());

		m_num_features=p->get_num_vectors();
	}
}

CNOCCO::~CNOCCO()
{
}

void CNOCCO::init()
{
	SG_ADD(&m_num_features, "num_features",
			"Number of features from each of the distributions",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "epsilon", "The regularization constant",
			MS_NOT_AVAILABLE);

	m_num_features=0;
	m_epsilon=0.0;

	// we need PERMUTATION as the null approximation method here
	m_null_approximation_method=PERMUTATION;
}

void CNOCCO::set_p(CFeatures* p)
{
	CIndependenceTest::set_p(p);
	REQUIRE(m_p, "Provided feature for p cannot be null!\n");
	m_num_features=m_p->get_num_vectors();
}

void CNOCCO::set_q(CFeatures* q)
{
	CIndependenceTest::set_q(q);
	REQUIRE(m_q, "Provided feature for q cannot be null!\n");
	m_num_features=m_q->get_num_vectors();
}

void CNOCCO::set_epsilon(float64_t epsilon)
{
	m_epsilon=epsilon;
}

float64_t CNOCCO::get_epsilon() const
{
	return m_epsilon;
}

SGMatrix<float64_t> CNOCCO::compute_helper(SGMatrix<float64_t> m)
{
	SG_DEBUG("Entering!\n");

	const index_t n=m_num_features;
	Map<MatrixXd> mat(m.matrix, n, n);

	// the result matrix res = m * inv(m + n*epsilon*eye(n,n))
	SGMatrix<float64_t> res(n, n);
	MatrixXd to_inv=mat+n*m_epsilon*MatrixXd::Identity(n,n);

	// since the matrix is SPD, instead of directly computing the inverse,
	// we compute the Cholesky decomposition and solve systems (see class
	// documentation for details)
	LLT<MatrixXd> chol(to_inv);

	// compute the matrix times inverse by solving systems
	VectorXd e=VectorXd::Zero(n);
	for (index_t i=0; i<n; ++i)
	{
		e(i)=1;

		// the solution vector corresponds to the i-th column of the inverse
		const VectorXd& x=chol.solve(e);
#pragma omp parallel for shared (res, mat, x, i)
		for (index_t j=0; j<n; ++j)
		{
			// since mat is symmetric we can use mat.col instead of mat.row here
			// for faster execution since matrices are column-major
			res(j,i)=x.dot(mat.col(j));
		}
		e(i)=0;
	}

	SG_DEBUG("Leaving!\n");

	return res;
}

float64_t CNOCCO::compute_statistic()
{
	SG_DEBUG("Entering!\n");

	REQUIRE(m_kernel_p, "Kernel for p is not set! Use set_kernel_p() method to "
			"set the kernel for use!\n");
	REQUIRE(m_kernel_q, "Kernel for q is not set! Use set_kernel_q() method to "
			"set the kernel for use!\n");

	REQUIRE(m_p && m_q, "features needed!\n")

	// compute kernel matrices
	SGMatrix<float64_t> Gx=get_kernel_matrix_K();
	SGMatrix<float64_t> Gy=get_kernel_matrix_L();

	// center the kernel matrices
	Gx.center();
	Gy.center();

	SGMatrix<float64_t> Rx=compute_helper(Gx);
	SGMatrix<float64_t> Ry=compute_helper(Gy);

	Map<MatrixXd> Rx_map(Rx.matrix, Rx.num_rows, Rx.num_cols);
	Map<MatrixXd> Ry_map(Ry.matrix, Ry.num_rows, Ry.num_cols);

	// compute the trace of the matrix multiplication without computing the
	// off-diagonal entries of the final matrix and just the diagonal entries
	float64_t result=0.0;
	for (index_t i=0; i<m_num_features; ++i)
	{
		// taking advantange of the symmetry, we can use Ry_map.col here
		// instead of Ry_map.row for computing the trace for computational
		// advantage since matrices are stored in column-major format
		result+=Rx_map.col(i).dot(Ry_map.col(i));
	}

	SG_DEBUG("leaving!\n");

	return result;
}

float64_t CNOCCO::compute_p_value(float64_t statistic)
{
	float64_t result=0;
	switch (m_null_approximation_method)
	{
	case PERMUTATION:
	{
		/* sampling null is handled there */
		result=CIndependenceTest::compute_p_value(statistic);
		break;
	}
	default:
		SG_ERROR("Use only PERMUTATION for null-approximation method "
				"for NOCCO!\n");
	}

	return result;
}

float64_t CNOCCO::compute_threshold(float64_t alpha)
{
	float64_t result=0;
	switch (m_null_approximation_method)
	{
	case PERMUTATION:
	{
		/* sampling null is handled there */
		result=CIndependenceTest::compute_threshold(alpha);
		break;
	}
	default:
		SG_ERROR("Use only PERMUTATION for null-approximation method "
				"for NOCCO!\n");
	}

	return result;
}

SGVector<float64_t> CNOCCO::sample_null()
{
	SG_DEBUG("Entering!\n")

	/* replace current kernel via precomputed custom kernel and call superclass
	 * method */

	/* backup references to old kernels */
	CKernel* kernel_p=m_kernel_p;
	CKernel* kernel_q=m_kernel_q;

	/* init kernels before to be sure that everything is fine
	 * kernel function between two samples from different distributions
	 * is never computed - in fact, they may as well have different features */
	m_kernel_p->init(m_p, m_p);
	m_kernel_q->init(m_q, m_q);

	/* precompute kernel matrices */
	CCustomKernel* precomputed_p=new CCustomKernel(m_kernel_p);
	CCustomKernel* precomputed_q=new CCustomKernel(m_kernel_q);
	SG_REF(precomputed_p);
	SG_REF(precomputed_q);

	/* temporarily replace own kernels */
	m_kernel_p=precomputed_p;
	m_kernel_q=precomputed_q;

	/* use superclass sample_null which shuffles the entries for one
	 * distribution using index permutation on rows and columns of
	 * kernel matrix from one distribution, while accessing the other
	 * in its original order and then compute statistic */
	SGVector<float64_t> null_samples=CKernelIndependenceTest::sample_null();

	/* restore kernels */
	m_kernel_p=kernel_p;
	m_kernel_q=kernel_q;

	SG_UNREF(precomputed_p);
	SG_UNREF(precomputed_q);

	SG_DEBUG("Leaving!\n")
	return null_samples;
}
