/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Eleftherios Avramidis
 */

#include "shogun/mathematics/Math.h"
#include <benchmark/benchmark.h>
#include "shogun/lib/SGVector.h"


namespace shogun
{

void BM_SGVector_calculation(benchmark::State& state)
{
	for (auto _ : state)
	{
		SGVector<float64_t> a(state.range(0)), b(state.range(0));
		a.random(0.0, 1E10);
		for (index_t i = 0; i < a.size(); ++i)
			b[i] = a[i] *2;
	}
}

BENCHMARK(BM_SGVector_calculation)->Range(8, 2048);

}
