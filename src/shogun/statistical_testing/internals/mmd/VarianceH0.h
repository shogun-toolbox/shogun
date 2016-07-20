/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 - 2016 Soumyajit De
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

#ifndef VARIANCE_H0__H_
#define VARIANCE_H0__H_

#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

template <typename T> class SGMatrix;

namespace internal
{

namespace mmd
{

struct VarianceH0
{
	template <typename T>
	T operator()(const SGMatrix<T>& kernel_matrix)
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;

		Eigen::Map<MatrixXt> map(kernel_matrix.matrix, kernel_matrix.num_rows, kernel_matrix.num_cols);
		index_t B=map.rows();

		VectorXt diag=map.diagonal();
		map.diagonal().setZero();

		auto term_1=CMath::sq(map.array().sum()/B/(B-1));
		auto term_2=map.array().square().sum()/B/(B-1);
		auto term_3=(map.colwise().sum()/(B-1)).array().square().sum()/B;

		map.diagonal()=diag;

		auto variance_estimate=2*(term_1+term_2-2*term_3);
		return variance_estimate;
	}
};

}

}

}

#endif // VARIANCE_H0__H_
