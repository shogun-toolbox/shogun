/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Soumyajit De
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
#include <shogun/mathematics/linalg/linalg.h>
#include <vector>
#include <algorithm>
#include <hayai/hayai.hpp>

using namespace shogun;

/**
 * Instructions :
 * 1. Install benchmarking toolkit "hayai" (https://github.com/nickbruun/hayai)
 * 2. Compile against libhayai_main, e.g.
 * 		g++ -O3 -std=c++11 matrix_product_benchmark.cpp -I/usr/include/eigen3 \
 *		-lshogun -lhayai_main -lOpenCL -o benchmark
 * 3. ./benchmark
 */

/** Generate data only once */
struct Data
{
	Data()
	{
		A = init(1);
		B = init(2);
		Av = init(A);
		Bv = init(B);
	}

	SGMatrix<float32_t> init(float32_t begin)
	{
		index_t n = num_rows * num_cols;
		std::vector<float32_t> mem(n);
		std::iota(mem.data(), mem.data() + n, begin);

		SGMatrix<float32_t> m(num_rows, num_cols);
		std::copy(m.matrix, m.matrix + n, mem.data());

		return m;
	}

	CGPUMatrix<float32_t> init(SGMatrix<float32_t> m)
	{
		CGPUMatrix<float32_t> mv(m);
		return mv;
	}

	SGMatrix<float32_t> A;
	SGMatrix<float32_t> B;
	CGPUMatrix<float32_t> Av;
	CGPUMatrix<float32_t> Bv;

	static constexpr index_t num_rows = 100;
	static constexpr index_t num_cols = 100;
};

BENCHMARK_P(SGMatrix, matrix_product_eigen3, 10, 1000,
		(SGMatrix<float32_t> A, SGMatrix<float32_t> B))
{
	SGMatrix<float32_t> C = linalg::matrix_product<linalg::Backend::EIGEN3>(A, B);
}

BENCHMARK_P(SGMatrix, matrix_product_viennacl, 10, 1000,
		(SGMatrix<float32_t> A, SGMatrix<float32_t> B))
{
	SGMatrix<float32_t> C = linalg::matrix_product<linalg::Backend::VIENNACL>(A, B);
}

BENCHMARK_P(CGPUMatrix, matrix_product_eigen3, 10, 1000,
		(CGPUMatrix<float32_t> A, CGPUMatrix<float32_t> B))
{
	CGPUMatrix<float32_t> C = linalg::matrix_product<linalg::Backend::EIGEN3>(A, B);
}

BENCHMARK_P(CGPUMatrix, matrix_product_viennacl, 10, 1000,
		(CGPUMatrix<float32_t> A, CGPUMatrix<float32_t> B))
{
	CGPUMatrix<float32_t> C = linalg::matrix_product<linalg::Backend::VIENNACL>(A, B);
}

Data data;

BENCHMARK_P_INSTANCE(SGMatrix, matrix_product_eigen3, (data.A, data.B));
BENCHMARK_P_INSTANCE(SGMatrix, matrix_product_viennacl, (data.A, data.B));
BENCHMARK_P_INSTANCE(CGPUMatrix, matrix_product_eigen3, (data.Av, data.Bv));
BENCHMARK_P_INSTANCE(CGPUMatrix, matrix_product_viennacl, (data.Av, data.Bv));
