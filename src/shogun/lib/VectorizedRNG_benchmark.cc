/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saatvik Shah
 */

#include <benchmark/benchmark.h>
#include <shogun/mathematics/Math.h>
#include "SGVector.h"
#include "SGMatrix.h"

namespace shogun
{
static void BM_SGVectorSimdRng(benchmark::State& state) {
    auto sg_vec = SGVector<float64_t>(state.range(0));
    for (auto _ : state) {
        sg_vec.random();
    }
    state.SetBytesProcessed(int64_t(state.iterations()) *
        int64_t(state.range(0)));
}
static void BM_SGVectorSimpleForRng(benchmark::State& state) {
    auto sg_vec = SGVector<float64_t>(state.range(0));
    CRandom r{};
    for (auto _ : state) {
        sg_vec.random(0, 1);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) *
        int64_t(state.range(0)));
}
BENCHMARK(BM_SGVectorSimpleForRng)->RangeMultiplier(4)->Range(256, 8<<20);
BENCHMARK(BM_SGVectorSimdRng)->RangeMultiplier(4)->Range(256, 8<<20);

static void BM_SGMatrixSimdRng(benchmark::State& state) {
    auto sg_mat = SGMatrix<float64_t>(state.range(0), state.range(0));
    for (auto _ : state) {
        sg_mat.random();
    }
    state.SetBytesProcessed(int64_t(state.iterations()) *
        int64_t(state.range(0)*state.range(0)));
}
static void BM_SGMatrixSimpleForRng(benchmark::State& state) {
    auto sg_mat = SGMatrix<float64_t>(state.range(0), state.range(0));
    for (auto _ : state) {
        std::transform(sg_mat.begin(), sg_mat.end(), sg_mat.begin(),
                       [](auto){ return CMath::random(static_cast<float64_t>(0),
                           static_cast<float64_t>(1)); });
    }
    state.SetBytesProcessed(int64_t(state.iterations()) *
        int64_t(state.range(0)*state.range(0)));
}
BENCHMARK(BM_SGMatrixSimpleForRng)->RangeMultiplier(2)->Range(64, 8<<10);
BENCHMARK(BM_SGMatrixSimdRng)->RangeMultiplier(2)->Range(64, 8<<10);
}
