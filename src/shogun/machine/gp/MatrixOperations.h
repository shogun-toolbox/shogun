/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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

#ifndef _MATRIXOPERATIONS_H_
#define _MATRIXOPERATIONS_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
template<class T> class SGVector;
template<class C> class SGMatrix;

/** @brief The helper class is used for Laplace and KL methods
 *
 */
class CMatrixOperations
{
public:
	static SGMatrix<float64_t> get_choleksy(SGVector<float64_t> W, SGVector<float64_t> sW,
		SGMatrix<float64_t> kernel, float64_t scale);

	static SGMatrix<float64_t> get_choleksy(Eigen::VectorXd eigen_W, Eigen::VectorXd eigen_sW,
		Eigen::MatrixXd eigen_kernel, float64_t scale);

	static SGMatrix<float64_t> get_inverse(SGMatrix<float64_t> L, SGMatrix<float64_t> kernel,
		SGVector<float64_t> sW, SGMatrix<float64_t> V, float64_t scale);

	static SGMatrix<float64_t> get_inverse(Eigen::MatrixXd eigen_L, Eigen::MatrixXd eigen_kernel,
		Eigen::VectorXd eigen_sW, Eigen::MatrixXd eigen_V, float64_t scale);

	static SGMatrix<float64_t> get_inverse(SGMatrix<float64_t> L, SGMatrix<float64_t> kernel,
		SGVector<float64_t> sW, float64_t scale);

	static SGMatrix<float64_t> get_inverse(Eigen::MatrixXd eigen_L, Eigen::MatrixXd eigen_kernel,
		Eigen::VectorXd eigen_sW, float64_t scale);

	static float64_t get_log_det(const Eigen::MatrixXd eigen_A);

	static float64_t get_log_det(const SGMatrix<float64_t> A);

};
}
#endif /* HAVE_EIGEN3 */
#endif /* _MATRIXOPERATIONS_H_ */
