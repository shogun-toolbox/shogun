#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/linalg/linalg.h>
#include <algorithm>
#include <hayai/hayai.hpp>
#include <iostream>

using namespace shogun;

/**
 * Instructions :
 * 1. Install benchmarking toolkit "hayai" (https://github.com/nickbruun/hayai)
 * 2. Compile against libhayai_main, e.g.
 * g++ -O3 -std=c++11 sgvector_add_operator_benchmark.cpp -I/usr/include/eigen3 -I/usr/local/include/viennacl -lshogun -lhayai_main -lOpenCL -o benchmark
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
		m_vec = SGVector<float32_t>(num_elems);
		std::iota(m_vec.data(), m_vec.data()+m_vec.size(), 1);
	}

	SGVector<float32_t> m_vec;
	static constexpr index_t num_elems=1000;
};

Data data;


BENCHMARK(SGVector, addoperator_SGVector, 10, 100000000)
{
	SGVector<float32_t> test_vec = SGVector<float32_t>(data.num_elems);
	std::iota(test_vec.data(), test_vec.data()+test_vec.size(), 1);
	data.m_vec += test_vec;
}

