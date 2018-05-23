/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef _DOTFEATURES_BENCHMARK_H_
#define _DOTFEATURES_BENCHMARK_H_

#define DOTFEATURES_BENCHMARK_DENSEDOT(FIXTURE, NAME)		\
BENCHMARK_DEFINE_F(FIXTURE, NAME)(benchmark::State& state)	\
{															\
	for (auto _ : state)									\
	{														\
		index_t dim = f->get_dim_feature_space();			\
		for (index_t i = 0; i < f->get_num_vectors(); ++i)	\
			f->dense_dot(i, w.vector, dim);					\
	}														\
}															\
BENCHMARK_REGISTER_F(FIXTURE, NAME)

#define DOTFEATURES_BENCHMARK_ADDDENSE(FIXTURE, NAME)		\
BENCHMARK_DEFINE_F(FIXTURE, NAME)(benchmark::State& state)	\
{															\
	for (auto _ : state)									\
	{														\
		index_t dim = f->get_dim_feature_space();			\
		for (index_t i = 0; i < f->get_num_vectors(); ++i)	\
			f->dense_dot(i, w.vector, dim);					\
	}														\
}															\
BENCHMARK_REGISTER_F(FIXTURE, NAME)


#endif /*  */
