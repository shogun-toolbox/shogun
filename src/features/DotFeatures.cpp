/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/DotFeatures.h"
#include "lib/io.h"
#include "base/Parallel.h"

#ifndef WIN32
#include <pthread.h>
#endif

struct DF_THREAD_PARAM
{
	CDotFeatures* df;
	float64_t* output;
	int32_t start;
	int32_t stop;
	float64_t* alphas;
	float64_t* vec;
	int32_t dim;
	float64_t bias;
	bool progress;
};

void CDotFeatures::dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
{
	ASSERT(output);
	ASSERT(start>=0);
	ASSERT(start<stop);
	ASSERT(stop<=get_num_vectors());

	int32_t num_vectors=stop-start;
	ASSERT(num_vectors>0);

	int32_t num_threads=parallel.get_num_threads();
	ASSERT(num_threads>0);

#ifndef WIN32
	if (num_threads < 2)
	{
#endif
		DF_THREAD_PARAM params;
		params.df=this;
		params.output=output;
		params.start=start;
		params.stop=stop;
		params.alphas=alphas;
		params.vec=vec;
		params.dim=dim;
		params.bias=b;
		params.progress=false; //true;
		dense_dot_range_helper((void*) &params);
#ifndef WIN32
	}
	else
	{
		pthread_t* threads = new pthread_t[num_threads-1];
		DF_THREAD_PARAM* params = new DF_THREAD_PARAM[num_threads];
		int32_t step= num_vectors/num_threads;

		int32_t t;

		for (t=0; t<num_threads-1; t++)
		{
			params[t].df = this;
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

		delete[] params;
		delete[] threads;
	}
#endif
}

void* CDotFeatures::dense_dot_range_helper(void* p)
{
	DF_THREAD_PARAM* par=(DF_THREAD_PARAM*) p;
	CDotFeatures* df=par->df;
	float64_t* output=par->output;
	int32_t start=par->start;
	int32_t stop=par->stop;
	float64_t* alphas=par->alphas;
	float64_t* vec=par->vec;
	int32_t dim=par->dim;
	float64_t bias=par->bias;
	bool progress=par->progress;

	if (alphas)
	{
		for (int32_t i=start; i<stop; i++)
		{
			output[i]=alphas[i]*df->dense_dot(i, vec, dim)+bias;
			if (progress)
				df->display_progress(start, stop, i);
		}
	}
	else
	{
		for (int32_t i=start; i<stop; i++)
		{
			output[i]=df->dense_dot(i, vec, dim)+bias;
			if (progress)
				df->display_progress(start, stop, i);
		}
	}

	return NULL;
}

void CDotFeatures::get_feature_matrix(float64_t** matrix, int32_t* d1, int32_t* d2)
{
    int64_t offs=0;
	int32_t num_vec=get_num_vectors();
    int32_t dim=get_dim_feature_space();
    ASSERT(num_vec>0);
    ASSERT(dim>0);

    int64_t sz=((uint64_t) num_vec)* dim;

    *d1=dim;
    *d2=num_vec;
    *matrix=new float64_t[sz];
    memset(*matrix, 0, sz*sizeof(float64_t));

    for (int32_t i=0; i<num_vec; i++)
    {
		add_to_dense_vec(1.0, i, &((*matrix)[offs]), dim);
        offs+=dim;
    }
}
