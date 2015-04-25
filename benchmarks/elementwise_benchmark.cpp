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
 * g++ -O3 -std=c++11 elementwise_benchmark.cpp -I/usr/include/eigen3 -lshogun -lhayai_main -lOpenCL -o benchmark
 * 3. ./benchmark
 */

/** Generate data only once */
struct Data
{
	Data()
	{
		init();
	}

	void init()
	{
		m_cpu=SGMatrix<float32_t>(num_rows, num_cols);
		std::iota(m_cpu.data(), m_cpu.data()+m_cpu.size(), 1);
		m_gpu=CGPUMatrix<float32_t>(m_cpu);
	}

	SGMatrix<float32_t> m_cpu;
	CGPUMatrix<float32_t> m_gpu;

	static constexpr index_t num_rows=1000;
	static constexpr index_t num_cols=1000;
};

Data data;

BENCHMARK(SGMatrix, elementwise, 10, 1000)
{
	float32_t weights=0.6;
	float32_t std_dev=0.2;
	float32_t mean=0.01;

	SGMatrix<float32_t> result=linalg::elementwise_compute(data.m_cpu,
	[&weights, &std_dev, &mean](float32_t& sqr_dist)
	{
		float32_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
		float32_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::sq(std_dev));
		float32_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
		return outer_factor*exp_factor*sin_factor;
	});

}

BENCHMARK(SGMatrix, loop, 10, 1000)
{
	float32_t weights=0.6;
	float32_t std_dev=0.2;
	float32_t mean=0.01;

	SGMatrix<float32_t> result(data.m_cpu.num_rows, data.m_cpu.num_cols);

	for (index_t j=0; j<data.m_cpu.num_cols; ++j)
	{
		for (index_t i=0; i<data.m_cpu.num_rows; ++i)
		{
			float32_t sqr_dist=data.m_cpu(i, j);
			float32_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
			float32_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::sq(std_dev));
			float32_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
			result(i, j)=outer_factor*exp_factor*sin_factor;
		}
	}
}

BENCHMARK(CGPUMatrix, elementwise, 10, 1000)
{
	float32_t weights=0.6;
	float32_t std_dev=0.2;
	float32_t mean=0.01;

	std::string data_type=linalg::implementation::ocl::get_type_string<float32_t>();

	std::string s_weights=std::to_string(weights);
	std::string s_std_dev=std::to_string(std_dev);
	std::string s_mean=std::to_string(mean);
	std::string s_pi=std::to_string(CMath::PI);

	std::string operation;
	operation.append(data_type+" outer_factor=-2*"+s_pi+"*sqrt(element)*pow("+s_weights+", 2);\n");
	operation.append(data_type+" exp_factor=exp(-2*pow("+s_pi+",2)*element*pow("+s_std_dev+", 2));\n");
	operation.append(data_type+" sin_factor=sin(2*"+s_pi+"*sqrt(element)*"+s_mean+");\n");
	operation.append("return outer_factor*exp_factor*sin_factor;");

	linalg::elementwise_compute_inplace(data.m_gpu, operation);
}
