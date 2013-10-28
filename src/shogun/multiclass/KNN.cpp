/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Written (W) 2011 Sergey Lisitsyn
 * Written (W) 2012 Fernando José Iglesias García, cover tree support
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/multiclass/KNN.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/JLCoverTree.h>
#include <shogun/lib/Time.h>
#include <shogun/base/Parameter.h>

//#define BENCHMARK_KNN
//#define DEBUG_KNN

using namespace shogun;

CKNN::CKNN()
: CDistanceMachine()
{
	init();
}

CKNN::CKNN(int32_t k, CDistance* d, CLabels* trainlab)
: CDistanceMachine()
{
	init();

	m_k=k;

	ASSERT(d)
	ASSERT(trainlab)

	set_distance(d);
	set_labels(trainlab);
	m_train_labels.vlen=trainlab->get_num_labels();
}

void CKNN::init()
{
	/* do not store model features by default (CDistanceMachine::apply(...) is
	 * overwritten */
	set_store_model_features(false);

	m_k=3;
	m_q=1.0;
	m_use_covertree=false;
	m_num_classes=0;

	/* use the method classify_multiply_k to experiment with different values
	 * of k */
	SG_ADD(&m_k, "m_k", "Parameter k", MS_NOT_AVAILABLE);
	SG_ADD(&m_q, "m_q", "Parameter q", MS_AVAILABLE);
	SG_ADD(&m_use_covertree, "m_use_covertree", "Parameter use_covertree", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_classes, "m_num_classes", "Number of classes", MS_NOT_AVAILABLE);
}

CKNN::~CKNN()
{
}

bool CKNN::train_machine(CFeatures* data)
{
	ASSERT(m_labels)
	ASSERT(distance)

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		distance->init(data, data);
	}

	SGVector<int32_t> lab=((CMulticlassLabels*) m_labels)->get_int_labels();
	m_train_labels.vlen=lab.vlen;
	m_train_labels.vector=SGVector<int32_t>::clone_vector(lab.vector, lab.vlen);
	ASSERT(m_train_labels.vlen>0)

	int32_t max_class=m_train_labels.vector[0];
	int32_t min_class=m_train_labels.vector[0];

	for (int32_t i=1; i<m_train_labels.vlen; i++)
	{
		max_class=CMath::max(max_class, m_train_labels.vector[i]);
		min_class=CMath::min(min_class, m_train_labels.vector[i]);
	}

	for (int32_t i=0; i<m_train_labels.vlen; i++)
		m_train_labels.vector[i]-=min_class;

	m_min_label=min_class;
	m_num_classes=max_class-min_class+1;

	SG_INFO("m_num_classes: %d (%+d to %+d) num_train: %d\n", m_num_classes,
			min_class, max_class, m_train_labels.vlen);

	return true;
}

SGMatrix<index_t> CKNN::nearest_neighbors()
{
	//number of examples to which kNN is applied
	int32_t n=distance->get_num_vec_rhs();
	//distances to train data
	float64_t* dists=SG_MALLOC(float64_t, m_train_labels.vlen);
	//indices to train data
	index_t* train_idxs=SG_MALLOC(index_t, m_train_labels.vlen);
	//pre-allocation of the nearest neighbors
	SGMatrix<index_t> NN(m_k, n);

	//for each test example
	for (int32_t i=0; i<n && (!CSignal::cancel_computations()); i++)
	{
		SG_PROGRESS(i, 0, n)

		//lhs idx 0..num train examples-1 (i.e., all train examples) and rhs idx i
		distances_lhs(dists,0,m_train_labels.vlen-1,i);

		//fill in an array with 0..num train examples-1
		for (int32_t j=0; j<m_train_labels.vlen; j++)
			train_idxs[j]=j;

		//sort the distance vector between test example i and all train examples
		CMath::qsort_index(dists, train_idxs, m_train_labels.vlen);

#ifdef DEBUG_KNN
		SG_PRINT("\nQuick sort query %d\n", i)
		for (int32_t j=0; j<m_k; j++)
			SG_PRINT("%d ", train_idxs[j])
		SG_PRINT("\n")
#endif

		//fill in the output the indices of the nearest neighbors
		for (int32_t j=0; j<m_k; j++)
			NN(j,i) = train_idxs[j];
	}

	SG_FREE(train_idxs);
	SG_FREE(dists);

	return NN;
}

CMulticlassLabels* CKNN::apply_multiclass(CFeatures* data)
{
	if (data)
		init_distance(data);

	//redirecting to fast (without sorting) classify if k==1
	if (m_k == 1)
		return classify_NN();

	ASSERT(m_num_classes>0)
	ASSERT(distance)
	ASSERT(distance->get_num_vec_rhs())

	int32_t num_lab=distance->get_num_vec_rhs();
	ASSERT(m_k<=distance->get_num_vec_lhs())

	CMulticlassLabels* output=new CMulticlassLabels(num_lab);

	//labels of the k nearest neighbors
	int32_t* train_lab=SG_MALLOC(int32_t, m_k);

	SG_INFO("%d test examples\n", num_lab)
	CSignal::clear_cancel();

	//histogram of classes and returned output
	float64_t* classes=SG_MALLOC(float64_t, m_num_classes);

#ifdef BENCHMARK_KNN
	CTime tstart;
	float64_t tfinish, tparsed, tcreated, tqueried;
#endif

	if ( ! m_use_covertree )
	{
		//get the k nearest neighbors of each example
		SGMatrix<index_t> NN = nearest_neighbors();

		//from the indices to the nearest neighbors, compute the class labels
		for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
		{
			//write the labels of the k nearest neighbors from theirs indices
			for (int32_t j=0; j<m_k; j++)
				train_lab[j] = m_train_labels[ NN(j,i) ];

			//get the index of the 'nearest' class
			int32_t out_idx = choose_class(classes, train_lab);
			//write the label of 'nearest' in the output
			output->set_label(i, out_idx + m_min_label);
		}

#ifdef BENCHMARK_KNN
		SG_PRINT(">>>> Quick sort applied in %9.4f\n",
				(tfinish = tstart.cur_time_diff(false)));
#endif
	}
	else	// Use cover tree
	{
		// m_q != 1.0 not supported with cover tree because the neighbors
		// are not retrieved in increasing order of distance to the query
		float64_t old_q = m_q;
		if ( old_q != 1.0 )
			SG_INFO("q != 1.0 not supported with cover tree, using q = 1\n")

		// From the sets of features (lhs and rhs) stored in distance,
		// build arrays of cover tree points
		v_array< CJLCoverTreePoint > set_of_points  =
			parse_points(distance, FC_LHS);
		v_array< CJLCoverTreePoint > set_of_queries =
			parse_points(distance, FC_RHS);

#ifdef BENCHMARK_KNN
		SG_PRINT(">>>> JL parsed in %9.4f\n",
			( tparsed = tstart.cur_time_diff(false) ) - tfinish);
#endif
		// Build the cover trees, one for the test vectors (rhs features)
		// and another for the training vectors (lhs features)
		CFeatures* r = distance->replace_rhs( distance->get_lhs() );
		node< CJLCoverTreePoint > top = batch_create(set_of_points);
		CFeatures* l = distance->replace_lhs(r);
		distance->replace_rhs(r);
		node< CJLCoverTreePoint > top_query = batch_create(set_of_queries);

#ifdef BENCHMARK_KNN
		SG_PRINT(">>>> Cover trees created in %9.4f\n",
				(tcreated = tstart.cur_time_diff(false)) - tparsed);
#endif

		// Get the k nearest neighbors to all the test vectors (batch method)
		distance->replace_lhs(l);
		v_array< v_array< CJLCoverTreePoint > > res;
		k_nearest_neighbor(top, top_query, res, m_k);

#ifdef BENCHMARK_KNN
		SG_PRINT(">>>> Query finished in %9.4f\n",
				(tqueried = tstart.cur_time_diff(false)) - tcreated);
#endif

#ifdef DEBUG_KNN
		SG_PRINT("\nJL Results:\n")
		for ( int32_t i = 0 ; i < res.index ; ++i )
		{
			for ( int32_t j = 0 ; j < res[i].index ; ++j )
			{
				printf("%d ", res[i][j].m_index);
			}
			printf("\n");
		}
		SG_PRINT("\n")
#endif

		for ( int32_t i = 0 ; i < res.index ; ++i )
		{
			// Translate from indices to labels of the nearest neighbors
			for ( int32_t j = 0; j < m_k; ++j )
				// The first index in res[i] points to the test vector
				train_lab[j] = m_train_labels.vector[ res[i][j+1].m_index ];

			// Get the index of the 'nearest' class
			int32_t out_idx = choose_class(classes, train_lab);
			output->set_label(res[i][0].m_index, out_idx+m_min_label);
		}

		m_q = old_q;

#ifdef BENCHMARK_KNN
		SG_PRINT(">>>> JL applied in %9.4f\n", tstart.cur_time_diff(false))
#endif
	}

	SG_FREE(classes);
	SG_FREE(train_lab);

	return output;
}

CMulticlassLabels* CKNN::classify_NN()
{
	ASSERT(distance)
	ASSERT(m_num_classes>0)

	int32_t num_lab = distance->get_num_vec_rhs();
	ASSERT(num_lab)

	CMulticlassLabels* output = new CMulticlassLabels(num_lab);
	float64_t* distances = SG_MALLOC(float64_t, m_train_labels.vlen);

	SG_INFO("%d test examples\n", num_lab)
	CSignal::clear_cancel();

	// for each test example
	for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		SG_PROGRESS(i,0,num_lab)

		// get distances from i-th test example to 0..num_m_train_labels-1 train examples
		distances_lhs(distances,0,m_train_labels.vlen-1,i);
		int32_t j;

		// assuming 0th train examples as nearest to i-th test example
		int32_t out_idx = 0;
		float64_t min_dist = distances[0];

		// searching for nearest neighbor by comparing distances
		for (j=0; j<m_train_labels.vlen; j++)
		{
			if (distances[j]<min_dist)
			{
				min_dist = distances[j];
				out_idx = j;
			}
		}

		// label i-th test example with label of nearest neighbor with out_idx index
		output->set_label(i,m_train_labels.vector[out_idx]+m_min_label);
	}

	SG_FREE(distances);
	return output;
}

SGMatrix<int32_t> CKNN::classify_for_multiple_k()
{
	ASSERT(m_num_classes>0)
	ASSERT(distance)
	ASSERT(distance->get_num_vec_rhs())

	int32_t num_lab=distance->get_num_vec_rhs();
	ASSERT(m_k<=num_lab)

	int32_t* output=SG_MALLOC(int32_t, m_k*num_lab);

	//working buffer of m_train_labels
	int32_t* train_lab=SG_MALLOC(int32_t, m_k);

	//histogram of classes and returned output
	int32_t* classes=SG_MALLOC(int32_t, m_num_classes);

	SG_INFO("%d test examples\n", num_lab)
	CSignal::clear_cancel();

	if ( ! m_use_covertree )
	{
		//get the k nearest neighbors of each example
		SGMatrix<index_t> NN = nearest_neighbors();

		for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
		{
			//write the labels of the k nearest neighbors from theirs indices
			for (int32_t j=0; j<m_k; j++)
				train_lab[j] = m_train_labels[ NN(j,i) ];

			choose_class_for_multiple_k(output+i, classes, train_lab, num_lab);
		}
	}
	else	// Use cover tree
	{
		//allocation for distances to nearest neighbors
		float64_t* dists=SG_MALLOC(float64_t, m_k);

		// From the sets of features (lhs and rhs) stored in distance,
		// build arrays of cover tree points
		v_array< CJLCoverTreePoint > set_of_points  =
			parse_points(distance, FC_LHS);
		v_array< CJLCoverTreePoint > set_of_queries =
			parse_points(distance, FC_RHS);

		// Build the cover trees, one for the test vectors (rhs features)
		// and another for the training vectors (lhs features)
		CFeatures* r = distance->replace_rhs( distance->get_lhs() );
		node< CJLCoverTreePoint > top = batch_create(set_of_points);
		CFeatures* l = distance->replace_lhs(r);
		distance->replace_rhs(r);
		node< CJLCoverTreePoint > top_query = batch_create(set_of_queries);

		// Get the k nearest neighbors to all the test vectors (batch method)
		distance->replace_lhs(l);
		v_array< v_array< CJLCoverTreePoint > > res;
		k_nearest_neighbor(top, top_query, res, m_k);

		for ( int32_t i = 0 ; i < res.index ; ++i )
		{
			// Handle the fact that cover tree doesn't return neighbors
			// ordered by distance

			for ( int32_t j = 0 ; j < m_k ; ++j )
			{
				// The first index in res[i] points to the test vector
				dists[j]     = distance->distance(res[i][j+1].m_index,
							res[i][0].m_index);
				train_lab[j] = m_train_labels.vector[
							res[i][j+1].m_index ];
			}

			// Now we get the indices to the neighbors sorted by distance
			CMath::qsort_index(dists, train_lab, m_k);

			choose_class_for_multiple_k(output+res[i][0].m_index, classes,
					train_lab, num_lab);
		}

		SG_FREE(dists);
	}

	SG_FREE(train_lab);
	SG_FREE(classes);

	return SGMatrix<int32_t>(output,num_lab,m_k,true);
}

void CKNN::init_distance(CFeatures* data)
{
	if (!distance)
		SG_ERROR("No distance assigned!\n")
	CFeatures* lhs=distance->get_lhs();
	if (!lhs || !lhs->get_num_vectors())
	{
		SG_UNREF(lhs);
		SG_ERROR("No vectors on left hand side\n")
	}
	distance->init(lhs, data);
	SG_UNREF(lhs);
}

bool CKNN::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CKNN::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

void CKNN::store_model_features()
{
	CFeatures* d_lhs=distance->get_lhs();
	CFeatures* d_rhs=distance->get_rhs();

	/* copy lhs of underlying distance */
	distance->init(d_lhs->duplicate(), d_rhs);

	SG_UNREF(d_lhs);
	SG_UNREF(d_rhs);
}

int32_t CKNN::choose_class(float64_t* classes, int32_t* train_lab)
{
	memset(classes, 0, sizeof(float64_t)*m_num_classes);

	float64_t multiplier = m_q;
	for (int32_t j=0; j<m_k; j++)
	{
		classes[train_lab[j]]+= multiplier;
		multiplier*= multiplier;
	}

	//choose the class that got 'outputted' most often
	int32_t out_idx=0;
	float64_t out_max=0;

	for (int32_t j=0; j<m_num_classes; j++)
	{
		if (out_max< classes[j])
		{
			out_idx= j;
			out_max= classes[j];
		}
	}

	return out_idx;
}

void CKNN::choose_class_for_multiple_k(int32_t* output, int32_t* classes, int32_t* train_lab, int32_t step)
{
	//compute histogram of class outputs of the first k nearest neighbours
	memset(classes, 0, sizeof(int32_t)*m_num_classes);

	for (int32_t j=0; j<m_k; j++)
	{
		classes[train_lab[j]]++;

		//choose the class that got 'outputted' most often
		int32_t out_idx=0;
		int32_t out_max=0;

		for (int32_t c=0; c<m_num_classes; c++)
		{
			if (out_max< classes[c])
			{
				out_idx= c;
				out_max= classes[c];
			}
		}

		output[j*step]=out_idx+m_min_label;
	}
}
