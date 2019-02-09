/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Eleftherios Avramidis
 */

#include "shogun/mathematics/Math.h"
#include <benchmark/benchmark.h>
#include "shogun/lib/SGMatrix.h"


namespace shogun
{

static SGMatrix<float64_t> createRandomData(const benchmark::State& state)
{
	index_t num_dim = state.range(0);
	index_t num_vecs = state.range(0);

	SGMatrix<float64_t> mat(num_dim, num_vecs);
	for (index_t i=0; i<num_vecs; i++)
	{
		for (index_t j=0; j<num_dim; j++)
		{
			mat(j,i) = CMath::random(0,1) + 0.5;
		}
	}
	return mat;
}

void BM_SGMatrix_eigenvectors(benchmark::State& state)
{
	for (auto _ : state)
	{
        SGMatrix<float64_t> mat = createRandomData(state);
        SGVector<float64_t> eigenvectors = SGMatrix<float64_t>::compute_eigenvectors(mat);
	}
}

void BM_SGMatrix_inverse(benchmark::State& state)
{
	for (auto _ : state)
	{
        SGMatrix<float64_t> mat = createRandomData(state);
        SGMatrix<float64_t>::inverse(mat);
	}
}

BENCHMARK(BM_SGMatrix_eigenvectors)->Range(8, 2048);
BENCHMARK(BM_SGMatrix_inverse)->Range(8, 2048);

}
