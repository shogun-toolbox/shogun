/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <benchmark/benchmark.h>

#include "shogun/features/DotFeatures_benchmark.h"
#include "shogun/features/RandomFourierDotFeatures.h"

namespace shogun
{

static std::shared_ptr<CRandomFourierDotFeatures> createRandomData(const benchmark::State& state)
{
	index_t num_dim = state.range(0);
	index_t num_vecs = 10000;

	SGMatrix<float64_t> mat(num_dim, num_vecs);
	for (index_t i=0; i<num_vecs; i++)
	{
		for (index_t j=0; j<num_dim; j++)
		{
			mat(j,i) = CMath::random(0,1) + 0.5;
		}
	}
	auto dense_feats = new CDenseFeatures<float64_t>(mat);
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

DOTFEATURES_BENCHMARK_DENSEDOT(RFFixture, RandomFourierDotFeatures_DenseDot)
DOTFEATURES_BENCHMARK_ADDDENSE(RFFixture, RandomFourierDotFeatures_AddDense)

BENCHMARK_REGISTER_F(RFFixture, RandomFourierDotFeatures_DenseDot)
	->RangeMultiplier(2)
	->Ranges({{128, 512}, {64, 512}})
	->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(RFFixture, RandomFourierDotFeatures_AddDense)
	->RangeMultiplier(2)
	->Ranges({{128, 512}, {64, 512}})
	->Unit(benchmark::kMillisecond);
}
