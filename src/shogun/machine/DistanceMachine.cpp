/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Christian Gehl
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST
 */

#include <shogun/machine/DistanceMachine.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct D_THREAD_PARAM
{
    CDistance* d;
    float64_t* r;
    int32_t idx_r_start;
    int32_t idx_start;
    int32_t idx_stop;
    int32_t idx_comp;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

CDistanceMachine::CDistanceMachine()
: CMachine()
{
	init();
}

CDistanceMachine::~CDistanceMachine()
{
	SG_UNREF(distance);
}

void CDistanceMachine::init()
{
	/* all distance machines should store their models, i.e. cluster centers
	 * At least, it has to be ensured, that after calling train(), or in the
	 * call of apply() in the cases where there is no train method, the lhs
	 * of the underlying distance is set to cluster centers */
	set_store_model_features(true);

	distance=NULL;
	m_parameters->add((CSGObject**)&distance, "distance", "Distance to use");
}

void CDistanceMachine::distances_lhs(float64_t* result,int32_t idx_a1,int32_t idx_a2,int32_t idx_b)
{
    int32_t num_threads=parallel->get_num_threads();
    ASSERT(num_threads>0)

    ASSERT(result)

    if (num_threads < 2)
    {
        D_THREAD_PARAM param;
        param.d=distance;
        param.r=result;
        param.idx_r_start=idx_a1;
        param.idx_start=idx_a1;
        param.idx_stop=idx_a2+1;
        param.idx_comp=idx_b;

        run_distance_thread_lhs((void*) &param);
    }
#ifdef HAVE_PTHREAD
    else
    {
        pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
        D_THREAD_PARAM* params = SG_MALLOC(D_THREAD_PARAM, num_threads);
        int32_t num_vec=idx_a2-idx_a1+1;
        int32_t step= num_vec/num_threads;
        int32_t t;

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        for (t=0; t<num_threads-1; t++)
        {
            params[t].d = distance;
            params[t].r = result;
            params[t].idx_r_start=t*step;
            params[t].idx_start = (t*step)+idx_a1;
            params[t].idx_stop = ((t+1)*step)+idx_a1;
            params[t].idx_comp=idx_b;

            pthread_create(&threads[t], &attr, CDistanceMachine::run_distance_thread_lhs, (void*)&params[t]);
        }
        params[t].d = distance;
        params[t].r = result;
        params[t].idx_r_start=t*step;
        params[t].idx_start = (t*step)+idx_a1;
        params[t].idx_stop = idx_a2+1;
        params[t].idx_comp=idx_b;

        run_distance_thread_lhs(&params[t]);

        for (t=0; t<num_threads-1; t++)
            pthread_join(threads[t], NULL);

        pthread_attr_destroy(&attr);
        SG_FREE(params);
        SG_FREE(threads);
    }
#endif
}

void CDistanceMachine::distances_rhs(float64_t* result,int32_t idx_b1,int32_t idx_b2,int32_t idx_a)
{
    int32_t num_threads=parallel->get_num_threads();
    ASSERT(num_threads>0)

    ASSERT(result)

    if (num_threads < 2)
    {
        D_THREAD_PARAM param;
        param.d=distance;
        param.r=result;
        param.idx_r_start=idx_b1;
        param.idx_start=idx_b1;
        param.idx_stop=idx_b2+1;
        param.idx_comp=idx_a;

        run_distance_thread_rhs((void*) &param);
    }
#ifndef WIN32
    else
    {
        pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
        D_THREAD_PARAM* params = SG_MALLOC(D_THREAD_PARAM, num_threads);
        int32_t num_vec=idx_b2-idx_b1+1;
        int32_t step= num_vec/num_threads;
        int32_t t;

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        for (t=0; t<num_threads-1; t++)
        {
            params[t].d = distance;
            params[t].r = result;
            params[t].idx_r_start=t*step;
            params[t].idx_start = (t*step)+idx_b1;
            params[t].idx_stop = ((t+1)*step)+idx_b1;
            params[t].idx_comp=idx_a;

            pthread_create(&threads[t], &attr, CDistanceMachine::run_distance_thread_rhs, (void*)&params[t]);
        }
        params[t].d = distance;
        params[t].r = result;
        params[t].idx_r_start=t*step;
        params[t].idx_start = (t*step)+idx_b1;
        params[t].idx_stop = idx_b2+1;
        params[t].idx_comp=idx_a;

        run_distance_thread_rhs(&params[t]);

        for (t=0; t<num_threads-1; t++)
            pthread_join(threads[t], NULL);

        pthread_attr_destroy(&attr);
        SG_FREE(params);
        SG_FREE(threads);
    }
#endif
}

void* CDistanceMachine::run_distance_thread_lhs(void* p)
{
    D_THREAD_PARAM* params= (D_THREAD_PARAM*) p;
    CDistance* distance=params->d;
    float64_t* res=params->r;
    int32_t idx_res_start=params->idx_r_start;
    int32_t idx_act=params->idx_start;
    int32_t idx_stop=params->idx_stop;
    int32_t idx_c=params->idx_comp;

    for (int32_t i=idx_res_start; idx_act<idx_stop; i++,idx_act++)
        res[i] =distance->distance(idx_act,idx_c);

    return NULL;
}

void* CDistanceMachine::run_distance_thread_rhs(void* p)
{
    D_THREAD_PARAM* params= (D_THREAD_PARAM*) p;
    CDistance* distance=params->d;
    float64_t* res=params->r;
    int32_t idx_res_start=params->idx_r_start;
    int32_t idx_act=params->idx_start;
    int32_t idx_stop=params->idx_stop;
    int32_t idx_c=params->idx_comp;

    for (int32_t i=idx_res_start; idx_act<idx_stop; i++,idx_act++)
        res[i] =distance->distance(idx_c,idx_act);

    return NULL;
}

CMulticlassLabels* CDistanceMachine::apply_multiclass(CFeatures* data)
{
	if (data)
	{
		/* set distance features to given ones and apply to all */
		CFeatures* lhs=distance->get_lhs();
		distance->init(lhs, data);
		SG_UNREF(lhs);

		/* build result labels and classify all elements of procedure */
		CMulticlassLabels* result=new CMulticlassLabels(data->get_num_vectors());
		for (index_t i=0; i<data->get_num_vectors(); ++i)
			result->set_label(i, apply_one(i));
		return result;
	}
	else
	{
		/* call apply on complete right hand side */
		CFeatures* all=distance->get_rhs();
		CMulticlassLabels* result = apply_multiclass(all);
		SG_UNREF(all);
		return result;
	}
	return NULL;
}

float64_t CDistanceMachine::apply_one(int32_t num)
{
	/* number of clusters */
	CFeatures* lhs=distance->get_lhs();
	int32_t num_clusters=lhs->get_num_vectors();
	SG_UNREF(lhs);

	/* (multiple threads) calculate distances to all cluster centers */
	float64_t* dists=SG_MALLOC(float64_t, num_clusters);
	distances_lhs(dists, 0, num_clusters-1, num);

	/* find cluster index with smallest distance */
	float64_t result=dists[0];
	index_t best_index=0;
	for (index_t i=1; i<num_clusters; ++i)
	{
		if (dists[i]<result)
		{
			result=dists[i];
			best_index=i;
		}
	}

	SG_FREE(dists);

	/* implicit cast */
	return best_index;
}

void CDistanceMachine::set_distance(CDistance* d)
{
	SG_REF(d);
	SG_UNREF(distance);
	distance=d;
}

CDistance* CDistanceMachine::get_distance() const
{
	SG_REF(distance);
	return distance;
}

void CDistanceMachine::store_model_features()
{
	SG_ERROR("store_model_features not yet implemented for %s!\n",
	         get_name());
}

