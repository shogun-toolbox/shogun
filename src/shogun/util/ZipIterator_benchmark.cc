/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <benchmark/benchmark.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/util/enumerate.h>

#include <random>
#include <iostream>

using namespace shogun;

SGMatrix<float64_t> createRandomData(const benchmark::State& state)
{
	std::random_device rd;
	std::mt19937_64 prng(rd());
	std::normal_distribution<float64_t> dist{0, 1};
 
	index_t num_dim = state.range(0);
	index_t num_vecs = state.range(0);

	SGMatrix<float64_t> mat(num_dim, num_vecs);
	for (index_t i=0; i<num_vecs; i++)
	{
		for (index_t j=0; j<num_dim; j++)
		{
			mat(j,i) = dist(prng);
		}
	}
	return mat;
}


void BM_without_zip_iterator(benchmark::State& state)
{
	for (auto _ : state)
	{
        const SGMatrix<float64_t> mat1 = createRandomData(state);
        const SGMatrix<float64_t> mat2 = createRandomData(state);

        auto result = SGMatrix<float64_t>(state.range(0), state.range(0));

        for (int64_t i = 0; i < result.size(); ++i)
        	result[i] = mat1[i] * mat2[i];
	}
}

void BM_with_zip_iterator(benchmark::State& state)
{
	for (auto _ : state)
	{
        const SGMatrix<float64_t> mat1 = createRandomData(state);
        const SGMatrix<float64_t> mat2 = createRandomData(state);

        auto result = SGMatrix<float64_t>(state.range(0), state.range(0));
        for (const auto& [idx, el1, el2]: enumerate(mat1, mat2))
    		result[idx] = el1 * el2;
	}
}

BENCHMARK(BM_without_zip_iterator)->Range(8, 2048);
BENCHMARK(BM_with_zip_iterator)->Range(8, 2048);