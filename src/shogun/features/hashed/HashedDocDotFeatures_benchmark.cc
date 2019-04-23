/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <benchmark/benchmark.h>

#include "shogun/features/DotFeatures_benchmark.h"
#include "shogun/features/hashed/HashedDocDotFeatures.h"
#include "shogun/lib/NGramTokenizer.h"
#include "shogun/mathematics/RandomNamespace.h"
#include "shogun/mathematics/UniformIntDistribution.h"
#include <random>
#include <vector>

namespace shogun
{

class HDFixture : public benchmark::Fixture
{
public:
	void SetUp(const ::benchmark::State& st)
	{
		std::random_device rd;
		std::mt19937_64 prng(rd);
		UniformIntDistribution<char> uniform_int_dist('A', 'Z');
		string_list.reserve(num_strings);
		for (index_t i=0; i<num_strings; i++)
		{
			string_list.push_back(SGVector<char>(max_str_length));
			for (index_t j=0; j<max_str_length; j++)
				string_list[i].vector[j] = (char) uniform_int_dist(prng);
		}
		auto string_feats = new CStringFeatures<char>(string_list, RAWBYTE);
		auto tzer = new NGramTokenizer(3);
		f = std::make_shared<HashedDocDotFeatures>(st.range(0), string_feats, tzer);

		w = SGVector<float64_t>(f->get_dim_feature_space());
		w.range_fill(17.0);
	}

	void TearDown(const ::benchmark::State&) { f.reset(); }

	std::shared_ptr<HashedDocDotFeatures> f;

	static constexpr index_t num_strings = 5000;
	static constexpr index_t max_str_length = 10000;
	std::vector<SGVector<char>> string_list;
	SGVector<float64_t> w;
};

#define ADD_HASHEDDOC_ARGS(WHAT)	\
	WHAT->Arg(8)->Arg(10)->Arg(12)->Arg(16)->Arg(20)->Unit(benchmark::kMillisecond)


ADD_HASHEDDOC_ARGS(DOTFEATURES_BENCHMARK_DENSEDOT(HDFixture, HashedDocDotFeatures_DenseDot));
ADD_HASHEDDOC_ARGS(DOTFEATURES_BENCHMARK_ADDDENSE(HDFixture, HashedDocDotFeatures_AddDense));
}
