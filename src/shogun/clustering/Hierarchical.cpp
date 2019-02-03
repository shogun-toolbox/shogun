/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann,
 *          Giovanni De Toni, Viktor Gal, Evan Shelhamer, Rukmangadh Sai Myana
 */

#include <iostream>
#include <limits>
#include <shogun/base/Parallel.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/progress.h>
#include <shogun/clustering/Hierarchical.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct pair
{
	/** index 1 */
	int32_t idx1;
	/** index 2 */
	int32_t idx2;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

CHierarchical::CHierarchical()
: CDistanceMachine()
{
	init();
	register_parameters();
}

CHierarchical::CHierarchical(int32_t merges_, CDistance* d)
: CDistanceMachine()
{
	init();
	merges = merges_;
	set_distance(d);
	register_parameters();
}

void CHierarchical::init()
{
	merges = 3;
	dimensions = 0;
	assignment = NULL;
	assignment_len = 0;
	table_size = 0;
	pairs = NULL;
	pairs_len = 0;
	merge_distance = NULL;
	merge_distance_len = 0;
}

void CHierarchical::register_parameters()
{
	watch_param("merges", &merges);
	watch_param("dimensions", &dimensions);
	watch_param("assignment", &assignment, &assignment_len);
	watch_param("table_size", &table_size);
	watch_param("pairs", &pairs, &pairs_len);
	watch_param("merge_distance", &merge_distance, &merge_distance_len);
}

CHierarchical::~CHierarchical()
{
	SG_FREE(merge_distance);
	SG_FREE(assignment);
	SG_FREE(pairs);
}

EMachineType CHierarchical::get_classifier_type()
{
	return CT_HIERARCHICAL;
}

bool CHierarchical::train_machine(CFeatures* data)
{
	ASSERT(distance)

	if (data)
		distance->init(data, data);

	CFeatures* lhs=distance->get_lhs();
	ASSERT(lhs)

	int32_t num=lhs->get_num_vectors();
	ASSERT(num>0)

	const int32_t num_pairs=num*(num-1)/2;

	SG_FREE(merge_distance);
	merge_distance=SG_MALLOC(float64_t, num);
	merge_distance_len=num;
	SGVector<float64_t>::fill_vector(merge_distance, num, -1.0);

	SG_FREE(assignment);
	assignment=SG_MALLOC(int32_t, num);
	assignment_len = num;
	SGVector<int32_t>::range_fill_vector(assignment, num);

	SG_FREE(pairs);
	pairs=SG_MALLOC(int32_t, 2*num);
	SGVector<int32_t>::fill_vector(pairs, 2*num, -1);

	pair* index=SG_MALLOC(pair, num_pairs);
	float64_t* distances=SG_MALLOC(float64_t, num_pairs);

	int32_t offs=0;
	for (auto i : SG_PROGRESS(range(0, num)))
	{
		for (int32_t j=i+1; j<num; j++)
		{
			distances[offs] = distance->distance(i, j);
			index[offs].idx1 = i;
			index[offs].idx2 = j;
			offs++; // offs=i*(i+1)/2+j
		}
	}

	CMath::qsort_index<float64_t,pair>(distances, index, (num-1)*num/2);
	//CMath::display_vector(distances, (num-1)*num/2, "dists");

	auto pb = SG_PROGRESS(range(0, num_pairs - 1));
	int32_t k=-1;
	int32_t l=0;
	for (; l < num && l < merges && k < num_pairs - 1; l++)
	{
		while (k<num_pairs-1)
		{
			k++;

			int32_t i=index[k].idx1;
			int32_t j=index[k].idx2;
			int32_t c1=assignment[i];
			int32_t c2=assignment[j];

			if (c1==c2)
				continue;

			pb.print_progress();

			if (c1<c2)
			{
				pairs[2*l]=c1;
				pairs[2*l+1]=c2;
			}
			else
			{
				pairs[2*l]=c2;
				pairs[2*l+1]=c1;
			}
			merge_distance[l]=distances[k];

			int32_t c=num+l;
			for (int32_t m=0; m<num; m++)
			{
				if (assignment[m] == c1 || assignment[m] == c2)
					assignment[m] = c;
			}
#ifdef DEBUG_HIERARCHICAL
			SG_PRINT("l=%04i i=%04i j=%04i c1=%+04d c2=%+04d c=%+04d dist=%6.6f\n", l,i,j, c1,c2,c, merge_distance[l])
#endif
			break;
		}
	}
	pb.complete();

	table_size=l-1;
	ASSERT(table_size>0)
	SG_FREE(distances);
	SG_FREE(index);
	SG_UNREF(lhs)

	return true;
}

bool CHierarchical::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CHierarchical::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}


int32_t CHierarchical::get_merges()
{
	return merges;
}

SGVector<int32_t> CHierarchical::get_assignment()
{
	return SGVector<int32_t>(assignment,table_size, false);
}

SGVector<float64_t> CHierarchical::get_merge_distances()
{
	return SGVector<float64_t>(merge_distance,merges, false);
}

SGMatrix<int32_t> CHierarchical::get_cluster_pairs()
{
	return SGMatrix<int32_t>(pairs,2,merges, false);
}

void CHierarchical::store_model_features()
{
	CDenseFeatures<float64_t>* rhs =
	    dynamic_cast<CDenseFeatures<float64_t>*>(distance->get_rhs());

	int32_t num_vectors = rhs->get_num_vectors();
	int32_t num_features = rhs->get_num_features();

	int32_t* n_cluster_samples = SG_MALLOC(int32_t, num_vectors + merges);
	SGVector<int32_t>::fill_vector(n_cluster_samples, num_vectors + merges, 0);

	SGMatrix<float64_t>* null_matrix =
	    new SGMatrix<float64_t>(num_features, num_vectors + merges);
	CDenseFeatures<float64_t>* temp_cluster_centers =
	    new CDenseFeatures<float64_t>(*null_matrix);
	delete null_matrix;

	for (int32_t i = 0; i < num_vectors; i++)
	{
		bool dofree_c, dofree_rhs;
		int32_t centerIdx = assignment[i];
#ifdef DEBUG_HIERARCHICAL
		SG_PRINT("\n");
		SG_PRINT("On %04i point, with assignment=%04i \n", i, centerIdx);
#endif
		float64_t* center = temp_cluster_centers->get_feature_vector(
		    centerIdx, num_features, dofree_c);
		float64_t* sample_point =
		    rhs->get_feature_vector(i, num_features, dofree_rhs);
		for (int32_t j = 0; j < num_features; j++)
		{
			if (n_cluster_samples[centerIdx] > 0)
			{
				center[j] = (center[j] * n_cluster_samples[centerIdx] +
				             sample_point[j]) /
				            (n_cluster_samples[centerIdx] + 1);
			}
			else
				center[j] = sample_point[j];
#ifdef DEBUG_HIERARCHICAL
			SG_PRINT(
			    "The %04i feature of the updated temp center is "
			    "%+06.6f \n",
			    j, center[j]);
#endif
		}
		n_cluster_samples[centerIdx]++;
		rhs->free_feature_vector(sample_point, num_features, dofree_rhs);
		temp_cluster_centers->free_feature_vector(
		    center, num_features, dofree_c);
	}

	null_matrix = new SGMatrix<float64_t>(num_features, num_vectors);
	CDenseFeatures<float64_t>* cluster_centers =
	    new CDenseFeatures<float64_t>(*null_matrix);
	delete null_matrix;

	int32_t curr_index = 0;
	for (int32_t i = 0; i < num_vectors + merges; i++)
	{
		if (n_cluster_samples[i] > 0)
		{
			bool dofree_temp, dofree;
			float64_t* temp_center = temp_cluster_centers->get_feature_vector(
			    i, num_features, dofree_temp);
			float64_t* center = cluster_centers->get_feature_vector(
			    curr_index, num_features, dofree);
			for (int32_t j = 0; j < num_features; j++)
#ifdef DEBUG_HIERARCHICAL
				SG_PRINT("\n");
			SG_PRINT("The %04i cluster center:\n", curr_index);
#endif
			{
				center[j] = temp_center[j];
#ifdef DEBUG_HIERARCHICAL
				SG_PRINT(
				    "The %04i feature of the center is %+06.6f \n", j,
				    center[j]);
#endif
			}
			curr_index++;
			temp_cluster_centers->free_feature_vector(
			    temp_center, num_features, dofree_temp);
			cluster_centers->free_feature_vector(center, num_features, dofree);
		}
	}
	distance->init(cluster_centers, rhs);
	SG_FREE(n_cluster_samples);
	SG_UNREF(temp_cluster_centers);
	SG_UNREF(rhs);
}
