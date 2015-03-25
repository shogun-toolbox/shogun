/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Yingrui Chang
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
 *
 */

#include <shogun/lib/config.h>
#include <iostream>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectDenseLeastSquareSolverQR.h>
#include <Eigen/LU>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

namespace shogun
{

CDirectDenseLeastSquareSolverQR::CDirectDenseLeastSquareSolverQR()
	: CLinearSolver<float64_t, float64_t>()
{
}

CDirectDenseLeastSquareSolverQR::~CDirectDenseLeastSquareSolverQR()
{
}

SGVector<float64_t> CDirectDenseLeastSquareSolverQR::solve(
		CLinearOperator<SGVector<float64_t>, SGVector<float64_t> >* A, SGVector<float64_t> b)
{
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension() == b.vlen, "Matrix dimension (%d) does not match vector dimension (%d)\n", A->get_dimension(), b.vlen);
	CDenseMatrixOperator<float64_t>* op
		=dynamic_cast<CDenseMatrixOperator<float64_t>*>(A);
	REQUIRE(op, "Operator \"%s\" is not of type DenseMatrixOperator.\n", op->get_name());

	// creating eigen3 Dense Matrix
	SGMatrix<float64_t> sm = op->get_matrix_operator();
	Map<MatrixXd> map_m(sm.matrix, sm.num_rows, sm.num_cols);

	// creating eigen3 maps for vectors
	SGVector<float64_t> x(sm.num_rows);
	x.set_const(0.0);
	Map<VectorXd> map_x(x.vector, x.vlen);
	Map<VectorXd> map_b(b.vector, b.vlen);
	
	// using Householder QR decomposition to find the least square solution to A^T x = b
	HouseholderQR<MatrixXd> hqr;
	hqr.compute(map_m.transpose());
	map_x = hqr.solve(map_b);

	return x;
}

}
#endif // HAVE_EIGEN3
