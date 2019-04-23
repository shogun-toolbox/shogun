/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein,
 *          Viktor Gal, Evan Shelhamer
 */

#include <shogun/machine/DistanceMachine.h>
#include <shogun/distance/Distance.h>
#include <shogun/base/Parameter.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

using namespace shogun;

DistanceMachine::DistanceMachine()
: Machine()
{
	init();
}

DistanceMachine::~DistanceMachine()
{

}

void DistanceMachine::init()
{
	/* all distance machines should store their models, i.e. cluster centers
	 * At least, it has to be ensured, that after calling train(), or in the
	 * call of apply() in the cases where there is no train method, the lhs
	 * of the underlying distance is set to cluster centers */
	set_store_model_features(true);

	distance=NULL;
	SG_ADD(&distance, "distance", "Distance to use", ParameterProperties::HYPER);
}

void DistanceMachine::distances_lhs(SGVector<float64_t>& result, index_t idx_a1, index_t idx_a2, index_t idx_b)
{
    int32_t num_threads;
    int32_t num_vec;
    int32_t step;

    ASSERT(result)

    #pragma omp parallel shared(num_threads, step)
    {
#ifdef HAVE_OPENMP
        #pragma omp single
        {
            num_threads=omp_get_num_threads();
            num_vec=idx_a2-idx_a1+1;
            step=num_vec/num_threads;
        }
        int32_t thread_num=omp_get_thread_num();

        index_t idx_r_start = thread_num * step;
        index_t idx_start = (thread_num * step) + idx_a1;
        index_t idx_stop = (thread_num==(num_threads - 1)) ? (idx_a2 + 1) : ((thread_num + 1) * step) + idx_a1;
        distance->run_distance_lhs(result, idx_r_start, idx_start, idx_stop, idx_b);
#else
        index_t idx_r_start = idx_a1;
        index_t idx_start = idx_a1;
        index_t idx_stop = idx_a2  + 1;
        distance->run_distance_lhs(result, idx_r_start, idx_start, idx_stop, idx_b);
#endif
    }
}

void DistanceMachine::distances_rhs(SGVector<float64_t>& result, index_t idx_b1, index_t idx_b2, index_t idx_a)
{
    int32_t num_threads;
    int32_t num_vec;
    int32_t step;

    ASSERT(result)

    #pragma omp parallel shared(num_threads, step)
    {
#ifdef HAVE_OPENMP
        #pragma omp single
        {
            num_threads=omp_get_num_threads();
            num_vec=idx_b2-idx_b1+1;
            step=num_vec/num_threads;
        }
        int32_t thread_num=omp_get_thread_num();

        index_t idx_r_start = thread_num * step;
        index_t idx_start = (thread_num * step) + idx_b1;
        index_t idx_stop = (thread_num==(num_threads - 1)) ? (idx_b2 + 1) : ((thread_num + 1) * step) + idx_b1;
        distance->run_distance_rhs(result, idx_r_start, idx_start, idx_stop, idx_a);
#else
        index_t idx_r_start = idx_b1;
        index_t idx_start = idx_b1;
        index_t idx_stop = idx_b2  + 1;
        distance->run_distance_rhs(result, idx_r_start, idx_start, idx_stop, idx_a);
#endif
    }
}

std::shared_ptr<MulticlassLabels> DistanceMachine::apply_multiclass(std::shared_ptr<Features> data)
{
	if (data)
	{
		/* set distance features to given ones and apply to all */
		auto lhs=distance->get_lhs();
		distance->init(lhs, data);

		/* build result labels and classify all elements of procedure */
		auto result=std::make_shared<MulticlassLabels>(data->get_num_vectors());
		for (index_t i=0; i<data->get_num_vectors(); ++i)
			result->set_label(i, apply_one(i));
		return result;
	}
	else
	{
		/* call apply on complete right hand side */
		auto all=distance->get_rhs();
		return apply_multiclass(all);
	}
	return NULL;
}

float64_t DistanceMachine::apply_one(int32_t num)
{
	/* number of clusters */
	auto lhs=distance->get_lhs();
	int32_t num_clusters=lhs->get_num_vectors();

	/* (multiple threads) calculate distances to all cluster centers */
    SGVector<float64_t> dists(num_clusters);
	distances_lhs(dists, 0, num_clusters-1, num);

	/* find cluster index with smallest distance */
	float64_t result=dists.vector[0];
	index_t best_index=0;
	for (index_t i=1; i<num_clusters; ++i)
	{
		if (dists[i]<result)
		{
			result=dists[i];
			best_index=i;
		}
	}

	/* implicit cast */
	return best_index;
}

void DistanceMachine::set_distance(std::shared_ptr<Distance> d)
{


	distance=d;
}

std::shared_ptr<Distance> DistanceMachine::get_distance() const
{

	return distance;
}

void DistanceMachine::store_model_features()
{
	SG_ERROR("store_model_features not yet implemented for %s!\n",
	         get_name());
}

