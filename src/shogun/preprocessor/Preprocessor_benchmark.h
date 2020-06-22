/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Nanubala Gnana Sai
 */

#ifndef _PREPROCESSOR_BENCHMARK_H_
#define _PREPROCESSOR_BENCHMARK_H_

namespace shogun
{

#define PREPROCESSOR_BENCHMARK_TRANSFORM(FIXTURE, NAME)                         \
BENCHMARK_DEFINE_F(FIXTURE, NAME)(benchmark::State& state)                      \
{                                                                               \
    index_t num_dims = state.range(0);                                          \
    index_t num_vecs = state.range(1);                                          \
                                                                                \
    SGMatrix<float64_t> matrix(num_dims, num_vecs);                             \
    linalg::range_fill(matrix, 1.0);                                            \
    auto feats = std::make_shared<DenseFeatures<float64_t>>(matrix);            \
    for(auto _ : state)                                                         \
    {                                                                           \
        preproc->transform(feats);        \
    }                                                                           \
}                                                                               \
BENCHMARK_REGISTER_F(FIXTURE, NAME)

#define PREPROCESSOR_BENCHMARK_FIT(FIXTURE, NAME)                         \
BENCHMARK_DEFINE_F(FIXTURE, NAME)(benchmark::State& state)                      \
{                                                                               \
    auto feats = std::make_shared<DenseFeatures<float64_t>>(mat);               \
    for(auto _ : state)                                                         \
    {                                                                           \
        preproc->fit(feats);        \
    }                                                                           \
}                                                                               \
BENCHMARK_REGISTER_F(FIXTURE, NAME)

}
#endif /* */