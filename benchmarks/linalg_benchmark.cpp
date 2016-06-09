/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 Pan Deng, Soumyajit De */

#include <shogun/base/init.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

#include <shogun/mathematics/eigen3.h>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/vector.hpp>

#include <algorithm>
#include <memory>
#include <hayai/hayai.hpp>

#include <iostream>

using namespace shogun;

/**
 * Instructions :
 * 1. Install benchmarking toolkit "hayai" (https://github.com/nickbruun/hayai)
 * 2. Compile against libhayai_main, e.g.
 * 		g++ -O3 -std=c++11 linglg_refactor_benchmark.cpp -I/usr/include/eigen3 \
 *		-lshogun -lhayai_main -lOpenCL -o benchmark
 * 3. ./benchmark
 */

/** Generate data only once */
typedef float32_t T;
typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;

template<typename value_type>
struct Data
{
	typedef viennacl::backend::mem_handle VCLMemoryArray;

	Data()
	{
		num_rows = 100;
		init();
	}

	Data(index_t num_rows)
	{
		this->num_rows = num_rows;
		init();
	}

	~Data()
	{
		exit_shogun();
	}

	void init()
	{
		A = init_sg(1);
		B = init_sg(2);
		Av = init_v(A);
		Bv = init_v(B);
		Ac = std::unique_ptr<BaseVector<value_type>> (init_c(A));
		Bc = std::unique_ptr<BaseVector<value_type>> (init_c(B));
		Ag = std::unique_ptr<BaseVector<value_type>> (init_g(A));
		Bg = std::unique_ptr<BaseVector<value_type>> (init_g(B));

		init_shogun_with_defaults();

		std::shared_ptr<GPUBackend> ViennaCLBackend;
	    ViennaCLBackend = std::shared_ptr<GPUBackend>(new GPUBackend);
	    sg_linalg->set_gpu_backend(ViennaCLBackend);
	}

	/** SGVector **/
	SGVector<value_type> init_sg(value_type begin)
	{
		SGVector<value_type> m(num_rows);
		m.range_fill(begin);

		return m;
	}

	/** ViennaCL Vector for test **/
	VCLVectorBase init_v(SGVector<value_type> m)
	{
		VCLVectorBase mv;
		std::shared_ptr<VCLMemoryArray> vector(new VCLMemoryArray());
		viennacl::backend::memory_create(*vector, sizeof(value_type)*num_rows,
			viennacl::context());
		viennacl::backend::memory_write(*vector, 0, num_rows*sizeof(value_type),
			m.vector);
		mv = VCLVectorBase(*vector, num_rows, 0, 1);

		return mv;
	}

	/** CPUVector derived from BaseVector **/
	std::unique_ptr<BaseVector<value_type>> init_c(SGVector<value_type> m)
	{
		std::unique_ptr<CPUVector<value_type>> mc(new CPUVector<value_type>(m));
        	return std::move(mc);
	}

	/** GPUVector derived from BaseVector **/
	std::unique_ptr<BaseVector<value_type>> init_g(SGVector<value_type> m)
	{
        	std::unique_ptr<GPUVector<value_type>> mg(new GPUVector<value_type>(m));
        	return std::move(mg);
	}

	SGVector<value_type> A;
	SGVector<value_type> B;
	VCLVectorBase Av;
	VCLVectorBase Bv;
	std::unique_ptr<BaseVector<value_type>> Ac;
	std::unique_ptr<BaseVector<value_type>> Bc;
	std::unique_ptr<BaseVector<value_type>> Ag;
	std::unique_ptr<BaseVector<value_type>> Bg;

	index_t num_rows;
};

BENCHMARK_P(CPUVector, dot_explict_eigen3, 10, 1000,
	(const SGVector<T> &A, const SGVector<T> &B))
{
	typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
	Eigen::Map<VectorXt> vec_A(A.vector, A.vlen);
	Eigen::Map<VectorXt> vec_B(B.vector, B.vlen);

    	auto C = vec_A.dot(vec_B);
}

BENCHMARK_P(CPUVector, dot_eigen3, 10, 1000,
	(BaseVector<T> *A, BaseVector<T> *B))
{
	auto C = sg_linalg->dot(A, B);
}

BENCHMARK_P(CPUVector_both_stack, dot_eigen3_cpu, 10, 1000,
	(const CPUVector<T> &A, const CPUVector<T> &B, CPUBackend &cpubackend))
{
	auto C = cpubackend.dot(A, B);
}

BENCHMARK_P(CPUVector_stack, dot_eigen3_cpu, 10, 1000,
	(CPUVector<T> *A, CPUVector<T> *B, CPUBackend &cpubackend))
{
	auto C = cpubackend.dot(*A, *B);
}

BENCHMARK_P(CPUVector_heap, dot_eigen3_cpu, 10, 1000,
	(CPUVector<T> *A, CPUVector<T> *B, CPUBackend *cpubackend))
{
	auto C = cpubackend->dot(*A, *B);
}

BENCHMARK_P(GPUVector, dot_explict_viennacl, 10, 1000,
	(const VCLVectorBase &A, const VCLVectorBase &B))
{
	auto C = viennacl::linalg::inner_prod(A, B);
}

BENCHMARK_P(GPUVector, dot_viennacl, 10, 1000,
	(BaseVector<T> *A, BaseVector<T> *B))
{
	auto C = sg_linalg->dot(A, B);
}

Data<T> data(1000000);
BENCHMARK_P_INSTANCE(CPUVector, dot_explict_eigen3, (data.A, data.B));
BENCHMARK_P_INSTANCE(CPUVector, dot_eigen3, (data.Ac.get(), data.Bc.get()));
BENCHMARK_P_INSTANCE(GPUVector, dot_explict_viennacl, (data.Av, data.Bv));
BENCHMARK_P_INSTANCE(GPUVector, dot_viennacl, (data.Ag.get(), data.Bg.get()));
