/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/Parameter.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>
#endif

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct DF_THREAD_PARAM
{
	CDotFeatures* df;
	int32_t* sub_index;
	float64_t* output;
	int32_t start;
	int32_t stop;
	float64_t* alphas;
	float64_t* vec;
	int32_t dim;
	float64_t bias;
	bool progress;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS


CDotFeatures::CDotFeatures(int32_t size)
	:CFeatures(size), combined_weight(1.0)
{
	init();
}


CDotFeatures::CDotFeatures(const CDotFeatures & orig)
	:CFeatures(orig), combined_weight(orig.combined_weight)
{
	init();
}


CDotFeatures::CDotFeatures(CFile* loader)
	:CFeatures(loader)
{
	init();
}

float64_t CDotFeatures::dense_dot_sgvec(int32_t vec_idx1, SGVector<float64_t> vec2)
{
	return dense_dot(vec_idx1, vec2.vector, vec2.vlen);
}

void CDotFeatures::dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
{
	ASSERT(output)
	// write access is internally between output[start..stop] so the following
	// line is necessary to write to output[0...(stop-start-1)]
	output-=start;
	ASSERT(start>=0)
	ASSERT(start<stop)
	ASSERT(stop<=get_num_vectors())

	int32_t num_vectors=stop-start;
	ASSERT(num_vectors>0)

	int32_t num_threads=parallel->get_num_threads();
	ASSERT(num_threads>0)

	CSignal::clear_cancel();

#ifdef HAVE_PTHREAD
	if (num_threads < 2)
	{
#endif
		DF_THREAD_PARAM params;
		params.df=this;
		params.sub_index=NULL;
		params.output=output;
		params.start=start;
		params.stop=stop;
		params.alphas=alphas;
		params.vec=vec;
		params.dim=dim;
		params.bias=b;
		params.progress=false; //true;
		dense_dot_range_helper((void*) &params);
#ifdef HAVE_PTHREAD
	}
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
		DF_THREAD_PARAM* params = SG_MALLOC(DF_THREAD_PARAM, num_threads);
		int32_t step= num_vectors/num_threads;

		int32_t t;

		for (t=0; t<num_threads-1; t++)
		{
			params[t].df = this;
			params[t].sub_index=NULL;
			params[t].output = output;
			params[t].start = start+t*step;
			params[t].stop = start+(t+1)*step;
			params[t].alphas=alphas;
			params[t].vec=vec;
			params[t].dim=dim;
			params[t].bias=b;
			params[t].progress = false;
			pthread_create(&threads[t], NULL,
					CDotFeatures::dense_dot_range_helper, (void*)&params[t]);
		}

		params[t].df = this;
		params[t].output = output;
		params[t].sub_index=NULL;
		params[t].start = start+t*step;
		params[t].stop = stop;
		params[t].alphas=alphas;
		params[t].vec=vec;
		params[t].dim=dim;
		params[t].bias=b;
		params[t].progress = false; //true;
		dense_dot_range_helper((void*) &params[t]);

		for (t=0; t<num_threads-1; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(params);
		SG_FREE(threads);
	}
#endif

#ifndef WIN32
		if ( CSignal::cancel_computations() )
			SG_INFO("prematurely stopped.           \n")
#endif
}

void CDotFeatures::dense_dot_range_subset(int32_t* sub_index, int32_t num, float64_t* output, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
{
	ASSERT(sub_index)
	ASSERT(output)

	int32_t num_threads=parallel->get_num_threads();
	ASSERT(num_threads>0)

	CSignal::clear_cancel();

#ifdef HAVE_PTHREAD
	if (num_threads < 2)
	{
#endif
		DF_THREAD_PARAM params;
		params.df=this;
		params.sub_index=sub_index;
		params.output=output;
		params.start=0;
		params.stop=num;
		params.alphas=alphas;
		params.vec=vec;
		params.dim=dim;
		params.bias=b;
		params.progress=false; //true;
		dense_dot_range_helper((void*) &params);
#ifdef HAVE_PTHREAD
	}
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
		DF_THREAD_PARAM* params = SG_MALLOC(DF_THREAD_PARAM, num_threads);
		int32_t step= num/num_threads;

		int32_t t;

		for (t=0; t<num_threads-1; t++)
		{
			params[t].df = this;
			params[t].sub_index=sub_index;
			params[t].output = output;
			params[t].start = t*step;
			params[t].stop = (t+1)*step;
			params[t].alphas=alphas;
			params[t].vec=vec;
			params[t].dim=dim;
			params[t].bias=b;
			params[t].progress = false;
			pthread_create(&threads[t], NULL,
					CDotFeatures::dense_dot_range_helper, (void*)&params[t]);
		}

		params[t].df = this;
		params[t].sub_index=sub_index;
		params[t].output = output;
		params[t].start = t*step;
		params[t].stop = num;
		params[t].alphas=alphas;
		params[t].vec=vec;
		params[t].dim=dim;
		params[t].bias=b;
		params[t].progress = false; //true;
		dense_dot_range_helper((void*) &params[t]);

		for (t=0; t<num_threads-1; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(params);
		SG_FREE(threads);
	}
#endif

#ifndef WIN32
		if ( CSignal::cancel_computations() )
			SG_INFO("prematurely stopped.           \n")
#endif
}

void* CDotFeatures::dense_dot_range_helper(void* p)
{
	DF_THREAD_PARAM* par=(DF_THREAD_PARAM*) p;
	CDotFeatures* df=par->df;
	int32_t* sub_index=par->sub_index;
	float64_t* output=par->output;
	int32_t start=par->start;
	int32_t stop=par->stop;
	float64_t* alphas=par->alphas;
	float64_t* vec=par->vec;
	int32_t dim=par->dim;
	float64_t bias=par->bias;
	bool progress=par->progress;

	if (sub_index)
	{
#ifdef WIN32
		for (int32_t i=start; i<stop i++)
#else
		for (int32_t i=start; i<stop &&
				!CSignal::cancel_computations(); i++)
#endif
		{
			if (alphas)
				output[i]=alphas[sub_index[i]]*df->dense_dot(sub_index[i], vec, dim)+bias;
			else
				output[i]=df->dense_dot(sub_index[i], vec, dim)+bias;
			if (progress)
				df->display_progress(start, stop, i);
		}

	}
	else
	{
#ifdef WIN32
		for (int32_t i=start; i<stop i++)
#else
		for (int32_t i=start; i<stop &&
				!CSignal::cancel_computations(); i++)
#endif
		{
			if (alphas)
				output[i]=alphas[i]*df->dense_dot(i, vec, dim)+bias;
			else
				output[i]=df->dense_dot(i, vec, dim)+bias;
			if (progress)
				df->display_progress(start, stop, i);
		}
	}

	return NULL;
}

SGMatrix<float64_t> CDotFeatures::get_computed_dot_feature_matrix()
{

	int64_t offs=0;
	int32_t num=get_num_vectors();
	int32_t dim=get_dim_feature_space();
	ASSERT(num>0)
	ASSERT(dim>0)

	SGMatrix<float64_t> m(dim, num);
	m.zero();

	for (int32_t i=0; i<num; i++)
	{
		add_to_dense_vec(1.0, i, &(m.matrix[offs]), dim);
		offs+=dim;
	}

	return m;
}

SGVector<float64_t> CDotFeatures::get_computed_dot_feature_vector(int32_t num)
{

	int32_t dim=get_dim_feature_space();
	ASSERT(num>=0 && num<=get_num_vectors())
	ASSERT(dim>0)

	SGVector<float64_t> v(dim);
	v.zero();
	add_to_dense_vec(1.0, num, v.vector, dim);
	return v;
}

void CDotFeatures::benchmark_add_to_dense_vector(int32_t repeats)
{
	int32_t num=get_num_vectors();
	int32_t d=get_dim_feature_space();
	float64_t* w= SG_MALLOC(float64_t, d);
	SGVector<float64_t>::fill_vector(w, d, 0.0);

	CTime t;
	float64_t start_cpu=t.get_runtime();
	float64_t start_wall=t.get_curtime();
	for (int32_t r=0; r<repeats; r++)
	{
		for (int32_t i=0; i<num; i++)
			add_to_dense_vec(1.172343*(r+1), i, w, d);
	}

	SG_PRINT("Time to process %d x num=%d add_to_dense_vector ops: cputime %fs walltime %fs\n",
			repeats, num, (t.get_runtime()-start_cpu)/repeats,
			(t.get_curtime()-start_wall)/repeats);

	SG_FREE(w);
}

void CDotFeatures::benchmark_dense_dot_range(int32_t repeats)
{
	int32_t num=get_num_vectors();
	int32_t d=get_dim_feature_space();
	float64_t* w= SG_MALLOC(float64_t, d);
	float64_t* out= SG_MALLOC(float64_t, num);
	float64_t* alphas= SG_MALLOC(float64_t, num);
#ifdef HAVE_LINALG_LIB
	llinalg::range_fill<linalg::Backend::NATIVE>(w, d, 17.0);
	linalg::range_fill<linalg::Backend::NATIVE>(alphas, num, 1.2345);
#else
	SGVector<float64_t>::range_fill_vector(w, d, 17.0);
	SGVector<float64_t>::range_fill_vector(alphas, num, 1.2345);
#endif

	CTime t;
	float64_t start_cpu=t.get_runtime();
	float64_t start_wall=t.get_curtime();

	for (int32_t r=0; r<repeats; r++)
			dense_dot_range(out, 0, num, alphas, w, d, 23);

#ifdef DEBUG_DOTFEATURES
	CMath::display_vector(out, 40, "dense_dot_range");
	float64_t* out2= SG_MALLOC(float64_t, num);

	for (int32_t r=0; r<repeats; r++)
	{
		CMath::fill_vector(out2, num, 0.0);
		for (int32_t i=0; i<num; i++)
			out2[i]+=dense_dot(i, w, d)*alphas[i]+23;
	}
	CMath::display_vector(out2, 40, "dense_dot");
	for (int32_t i=0; i<num; i++)
		out2[i]-=out[i];
	CMath::display_vector(out2, 40, "diff");
#endif
	SG_PRINT("Time to process %d x num=%d dense_dot_range ops: cputime %fs walltime %fs\n",
			repeats, num, (t.get_runtime()-start_cpu)/repeats,
			(t.get_curtime()-start_wall)/repeats);

	SG_FREE(alphas);
	SG_FREE(out);
	SG_FREE(w);
}

SGVector<float64_t> CDotFeatures::get_mean()
{
	int32_t num=get_num_vectors();
	int32_t dim=get_dim_feature_space();
	ASSERT(num>0)
	ASSERT(dim>0)

	SGVector<float64_t> mean(dim);
	memset(mean.vector, 0, sizeof(float64_t)*dim);

	for (int i = 0; i < num; i++)
		add_to_dense_vec(1, i, mean.vector, dim);
	for (int j = 0; j < dim; j++)
		mean.vector[j] /= num;

	return mean;
}

SGVector<float64_t> CDotFeatures::get_mean(CDotFeatures* lhs, CDotFeatures* rhs)
{
	ASSERT(lhs && rhs)
	ASSERT(lhs->get_dim_feature_space() == rhs->get_dim_feature_space())

	int32_t num_lhs=lhs->get_num_vectors();
	int32_t num_rhs=rhs->get_num_vectors();
	int32_t dim=lhs->get_dim_feature_space();
	ASSERT(num_lhs>0)
	ASSERT(num_rhs>0)
	ASSERT(dim>0)

	SGVector<float64_t> mean(dim);
	memset(mean.vector, 0, sizeof(float64_t)*dim);

	for (int i = 0; i < num_lhs; i++)
		lhs->add_to_dense_vec(1, i, mean.vector, dim);
	for (int i = 0; i < num_rhs; i++)
		rhs->add_to_dense_vec(1, i, mean.vector, dim);
	for (int j = 0; j < dim; j++)
		mean.vector[j] /= (num_lhs+num_rhs);

	return mean;
}

SGMatrix<float64_t> CDotFeatures::get_cov()
{
	int32_t num=get_num_vectors();
	int32_t dim=get_dim_feature_space();
	ASSERT(num>0)
	ASSERT(dim>0)

	SGMatrix<float64_t> cov(dim, dim);

	memset(cov.matrix, 0, sizeof(float64_t)*dim*dim);

	SGVector<float64_t> mean = get_mean();

	for (int i = 0; i < num; i++)
	{
		SGVector<float64_t> v = get_computed_dot_feature_vector(i);
		SGVector<float64_t>::add(v.vector, 1, v.vector, -1, mean.vector, v.vlen);
		for (int m = 0; m < v.vlen; m++)
		{
			for (int n = 0; n <= m ; n++)
			{
				(cov.matrix)[m*v.vlen+n] += v.vector[m]*v.vector[n];
			}
		}
	}
	for (int m = 0; m < dim; m++)
	{
		for (int n = 0; n <= m ; n++)
		{
			(cov.matrix)[m*dim+n] /= num;
		}
	}
	for (int m = 0; m < dim-1; m++)
	{
		for (int n = m+1; n < dim; n++)
		{
			(cov.matrix)[m*dim+n] = (cov.matrix)[n*dim+m];
		}
	}
	return cov;
}

SGMatrix<float64_t> CDotFeatures::compute_cov(CDotFeatures* lhs, CDotFeatures* rhs)
{
	CDotFeatures* feats[2];
	feats[0]=lhs;
	feats[1]=rhs;

	int32_t nums[2], dims[2], num=0;

	for (int i = 0; i < 2; i++)
	{
		nums[i]=feats[i]->get_num_vectors();
		dims[i]=feats[i]->get_dim_feature_space();
		ASSERT(nums[i]>0)
		ASSERT(dims[i]>0)
		num += nums[i];
	}

	ASSERT(dims[0]==dims[1])
	int32_t dim = dims[0];

	SGMatrix<float64_t> cov(dim, dim);

	memset(cov.matrix, 0, sizeof(float64_t)*dim*dim);

	SGVector<float64_t>  mean=get_mean(lhs,rhs);

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < nums[i]; j++)
		{
			SGVector<float64_t> v = feats[i]->get_computed_dot_feature_vector(j);
			SGVector<float64_t>::add(v.vector, 1, v.vector, -1, mean.vector, v.vlen);
			for (int m = 0; m < v.vlen; m++)
			{
				for (int n = 0; n <= m; n++)
				{
					(cov.matrix)[m*v.vlen+n] += v.vector[m]*v.vector[n];
				}
			}
		}
	}
	for (int m = 0; m < dim; m++)
	{
		for (int n = 0; n <= m; n++)
		{
			(cov.matrix)[m*dim+n] /= num;
		}
	}
	for (int m = 0; m < dim-1; m++)
	{
		for (int n = m+1; n < dim; n++)
		{
			(cov.matrix[m*dim+n]) = (cov.matrix)[n*dim+m];
		}
	}

	return cov;
}

void CDotFeatures::display_progress(int32_t start, int32_t stop, int32_t v)
{
	int32_t num_vectors=stop-start;
	int32_t i=v-start;

	if ( (i% (num_vectors/100+1))== 0)
		SG_PROGRESS(v, 0.0, num_vectors-1)
}

void CDotFeatures::init()
{
	set_property(FP_DOT);
	m_parameters->add(&combined_weight, "combined_weight",
					  "Feature weighting in combined dot features.");
}
