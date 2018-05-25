/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <benchmark/benchmark.h>

#include "shogun/mathematics/linalg/LinalgNamespace.h"

namespace shogun
{

template<typename T>
void BM_LinAlg_SGVector_zero(benchmark::State& state)
{
	for (auto _ : state)
	{
		SGVector<T> v(state.range(0));
		linalg::zero(v);
	}
}

template<typename T>
void BM_SGVector_zero(benchmark::State& state)
{
	for (auto _ : state)
	{
		SGVector<T> v(state.range(0));
		v.zero();
	}
}

BENCHMARK_TEMPLATE(BM_LinAlg_SGVector_zero, int32_t)->Range(8, 8<<10);
BENCHMARK_TEMPLATE(BM_LinAlg_SGVector_zero, int64_t)->Range(8, 8<<10);
BENCHMARK_TEMPLATE(BM_LinAlg_SGVector_zero, float32_t)->Range(8, 8<<10);
BENCHMARK_TEMPLATE(BM_LinAlg_SGVector_zero, float64_t)->Range(8, 8<<10);

BENCHMARK_TEMPLATE(BM_SGVector_zero, int32_t)->Range(8, 8<<10);
BENCHMARK_TEMPLATE(BM_SGVector_zero, int64_t)->Range(8, 8<<10);
BENCHMARK_TEMPLATE(BM_SGVector_zero, float32_t)->Range(8, 8<<10);
BENCHMARK_TEMPLATE(BM_SGVector_zero, float64_t)->Range(8, 8<<10);

}
