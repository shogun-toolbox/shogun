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

#ifndef DIRECT_DENSE_LEAST_SQUARE_SOLVER_QR_H
#define DIRECT_DENSE_LEAST_SQUARE_SOLVER_QR_H

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{

/** @brief Class that provides a solve method for finding the optimal least square solution
 * for linear system A*x=b using using QR decomposition.
 *
 * http://en.wikipedia.org/wiki/QR_decomposition
 *
 * Assumption: Inpute dense operator A needs to be full row rank.
 *
 */
class CDirectDenseLeastSquareSolverQR : public CLinearSolver<float64_t, float64_t>
{
public:
	/** Default constructor */
	CDirectDenseLeastSquareSolverQR();

	/** Destructor */
	virtual ~CDirectDenseLeastSquareSolverQR();

	/**
	 * Solve method for solving real-valued least square problem based on QR decomposition
	 *
	 * @param A the dense linear operator of the system, should be full row rank
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> solve(CLinearOperator<SGVector<float64_t>, SGVector<float64_t> >* A,
		SGVector<float64_t> b);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DirectDenseLeastSquareSolverQR";
	}

};

}

#endif // HAVE_EIGEN3
#endif // DIRECT_DENSE_LEAST_SQUARE_SOLVER_QR_H
