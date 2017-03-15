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

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

float32_t WithinBlockDirect::operator()(const SGMatrix<float32_t>& km)
{
	Eigen::Map<Eigen::MatrixXf> map(km.matrix, km.num_rows, km.num_cols);
	index_t B=km.num_rows;

	Eigen::VectorXf diag=map.diagonal();
	map.diagonal().setZero();

	auto term_1=map.array().square().sum();
	auto term_2=CMath::sq(map.array().sum());
	auto term_3=(map*map).array().sum();

	map.diagonal()=diag;

	auto variance_estimate=2*(term_1+term_2/(B-1)/(B-2)-2*term_3/(B-2))/B/(B-3);
	return variance_estimate;
}
