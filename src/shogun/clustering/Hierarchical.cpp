/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann,
 *          Giovanni De Toni, Viktor Gal, Evan Shelhamer
 */

#include <shogun/base/Parallel.h>
#include <shogun/base/progress.h>
#include <shogun/clustering/Hierarchical.h>
#include <shogun/distance/Distance.h>
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

Hierarchical::Hierarchical()
: DistanceMachine()
{
	init();
	register_parameters();
}

Hierarchical::Hierarchical(int32_t merges_, std::shared_ptr<Distance> d)
: DistanceMachine()
{
	init();
	merges = merges_;
	set_distance(d);
	register_parameters();
}

void Hierarchical::init()
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

void Hierarchical::register_parameters()
{
	watch_param("merges", &merges);
	watch_param("dimensions", &dimensions);
	watch_param("assignment", &assignment, &assignment_len);
	watch_param("table_size", &table_size);
	watch_param("pairs", &pairs, &pairs_len);
	watch_param("merge_distance", &merge_distance, &merge_distance_len);
}

Hierarchical::~Hierarchical()
{
	SG_FREE(merge_distance);
	SG_FREE(assignment);
	SG_FREE(pairs);
}

EMachineType Hierarchical::get_classifier_type()
{
	return CT_HIERARCHICAL;
}

bool Hierarchical::train_machine(std::shared_ptr<Features> data)
{
	ASSERT(distance)

	if (data)
		distance->init(data, data);

	auto lhs=distance->get_lhs();
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

	Math::qsort_index<float64_t,pair>(distances, index, (num-1)*num/2);
	//Math::display_vector(distances, (num-1)*num/2, "dists");

	auto pb = SG_PROGRESS(range(0, num_pairs - 1));
	int32_t k=-1;
	int32_t l=0;
	for (; l<num && (num-l)>=merges && k<num_pairs-1; l++)
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
			io::print("l={:04} i={:04d} j={:04d} c1={:+04} c2={:+04d} c={:+04d} dist={:6.6f}\n", l,i,j, c1,c2,c, merge_distance[l]);
#endif
			break;
		}
	}
	pb.complete();

	table_size=l-1;
	ASSERT(table_size>0)
	SG_FREE(distances);
	SG_FREE(index);


	return true;
}

bool Hierarchical::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool Hierarchical::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}


int32_t Hierarchical::get_merges()
{
	return merges;
}

SGVector<int32_t> Hierarchical::get_assignment()
{
	return SGVector<int32_t>(assignment,table_size, false);
}

SGVector<float64_t> Hierarchical::get_merge_distances()
{
	return SGVector<float64_t>(merge_distance,merges, false);
}

SGMatrix<int32_t> Hierarchical::get_cluster_pairs()
{
	return SGMatrix<int32_t>(pairs,2,merges, false);
}

