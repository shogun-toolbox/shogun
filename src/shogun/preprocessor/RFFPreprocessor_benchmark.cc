/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Nanubala Gnana Sai
 */

#include <benchmark/benchmark.h>
#include "shogun/preprocessor/RFFPreprocessor.h"
#include "shogun/mathematics/RandomNamespace.h"
#include "shogun/mathematics/linalg/LinalgNamespace.h"
#include <random>
#include "shogun/preprocessor/Preprocessor_benchmark.h"
#include "shogun/kernel/GaussianKernel.h"

namespace shogun
{

SGMatrix<float64_t> createRandomData(const benchmark::State& state)
{
	std::random_device rd;
	std::mt19937_64 prng(rd());
	UniformRealDistribution<float64_t> uniform(0, 1);
	
	index_t num_dims = state.range(0);
	index_t num_vecs = state.range(1);

	SGMatrix<float64_t> mat(num_dims, num_vecs);
	random::fill_array(mat, uniform, prng);

	return mat;
}

class TransformFixture: public benchmark::Fixture
{
	public:
		void SetUp(const ::benchmark::State& st)
		{
			auto mat = createRandomData(st);
			auto feats = std::make_shared<DenseFeatures<float64_t>>(mat);
			const index_t target_dim = st.range(2);
			auto gauss = std::make_shared<GaussianKernel>(width);

			preproc = std::make_shared<RFFPreprocessor>();
			preproc->set_kernel(gauss);
			preproc->set_dim_output(target_dim);
			preproc->fit(feats);
		}

		void TearDown(const ::benchmark::State&) {}

		std::shared_ptr<RFFPreprocessor> preproc;
		const float64_t width = 1.5;
};

class FitFixture: public benchmark::Fixture
{
	public:
		void SetUp(const ::benchmark::State& st)
		{
			mat = createRandomData(st);
			const index_t target_dim = st.range(2);

			auto gauss = std::make_shared<GaussianKernel>(width);

			preproc = std::make_shared<RFFPreprocessor>();
			preproc->set_kernel(gauss);
			preproc->set_dim_output(target_dim);
		}

		void TearDown(const ::benchmark::State&) {}

		std::shared_ptr<RFFPreprocessor> preproc;
		SGMatrix<float64_t> mat;
		const float64_t width = 1.5;
};

#define ADD_RFFPREPROCESSOR_ARGS(WHAT)	\
	WHAT->RangeMultiplier(10)->Ranges({{10,100}, {100,10000}, {100,10000}})->Unit(benchmark::kMillisecond);

ADD_RFFPREPROCESSOR_ARGS(PREPROCESSOR_BENCHMARK_TRANSFORM(TransformFixture, RFFPreprocessor_Transform))
ADD_RFFPREPROCESSOR_ARGS(PREPROCESSOR_BENCHMARK_FIT(FitFixture, RFFPreprocessor_Fit))

}

