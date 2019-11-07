/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <benchmark/benchmark.h>

#include "shogun/features/DotFeatures_benchmark.h"
#include "shogun/features/RandomFourierDotFeatures.h"
#include "shogun/mathematics/UniformIntDistribution.h"
#include <random>

namespace shogun
{

static std::shared_ptr<CRandomFourierDotFeatures> createRandomData(const benchmark::State& state)
{
	std::random_device rd;
	std::mt19937_64 prng(rd());
	UniformIntDistribution<index_t> uniform_int_dist(0, 1);
 
	index_t num_dim = state.range(0);
	index_t num_vecs = 10000;

	SGMatrix<float64_t> mat(num_dim, num_vecs);
	for (index_t i=0; i<num_vecs; i++)
	{
		for (index_t j=0; j<num_dim; j++)
		{
			mat(j,i) = uniform_int_dist(prng) + 0.5;
		}
	}
	auto dense_feats = new DenseFeatures<float64_t>(mat);
	SGVector<float64_t> params(1);
	params[0] = num_dim - 20;
	return std::make_shared<CRandomFourierDotFeatures>(dense_feats, state.range(1), KernelName::GAUSSIAN, params);
}

class RFFixture : public benchmark::Fixture
{
public:
	void SetUp(const ::benchmark::State& st)
	{
		f = createRandomData(st);
		w = SGVector<float64_t>(f->get_dim_feature_space());
		w.range_fill(17.0);
	}

	void TearDown(const ::benchmark::State&) { f.reset(); }

	std::shared_ptr<CRandomFourierDotFeatures> f;
	SGVector<float64_t> w;
};

#define ADD_RANDOMFOURIER_ARGS(WHAT)	\
	WHAT->RangeMultiplier(2)->Ranges({{128, 512}, {64, 512}})->Unit(benchmark::kMillisecond);

ADD_RANDOMFOURIER_ARGS(DOTFEATURES_BENCHMARK_DENSEDOT(RFFixture, RandomFourierDotFeatures_DenseDot))
ADD_RANDOMFOURIER_ARGS(DOTFEATURES_BENCHMARK_ADDDENSE(RFFixture, RandomFourierDotFeatures_AddDense))

}
