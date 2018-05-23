/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <benchmark/benchmark.h>

#include "shogun/lib/RefCount.h"

namespace shogun
{

static void BM_RefCount(benchmark::State& state)
{
	RefCount rf;
	for (auto _ : state)
		rf.ref();
}

BENCHMARK(BM_RefCount);

}
