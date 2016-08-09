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
#include <shogun/multiclass/tree/KDTree.h>
#include <shogun/lib/external/falconn/lsh_nn_table.h>
#include <shogun/mathematics/eigen3.h>
//#define DEBUG_KNN

using namespace shogun;
using namespace Eigen;

CKNN::CKNN()
: CDistanceMachine()
{
	init();
}

CKNN::CKNN(int32_t k, CDistance* d, CLabels* trainlab, KNN_SOLVER knn_solver)
: CDistanceMachine()
{
	init();

	m_k=k;

	ASSERT(d)
	ASSERT(trainlab)

	set_distance(d);
	set_labels(trainlab);
	m_train_labels.vlen=trainlab->get_num_labels();
	m_knn_solver=knn_solver;
}

void CKNN::init()
{
	/* do not store model features by default (CDistanceMachine::apply(...) is
	 * overwritten */
	set_store_model_features(false);

	m_k=3;
	m_q=1.0;
	m_num_classes=0;
	m_leaf_size=1;
	m_knn_solver=KNN_BRUTE;
	m_lsh_l = 0;
	m_lsh_t = 0;

	/* use the method classify_multiply_k to experiment with different values
	 * of k */
	SG_ADD(&m_k, "m_k", "Parameter k", MS_NOT_AVAILABLE);
	SG_ADD(&m_q, "m_q", "Parameter q", MS_AVAILABLE);
	SG_ADD(&m_num_classes, "m_num_classes", "Number of classes", MS_NOT_AVAILABLE);
	SG_ADD(&m_leaf_size, "m_leaf_size", "Leaf size for KDTree", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_knn_solver, "m_knn_solver", "Algorithm to solve knn", MS_NOT_AVAILABLE);
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
	m_train_labels=lab.clone();
	ASSERT(m_train_labels.vlen>0)

	int32_t max_class=m_train_labels[0];
	int32_t min_class=m_train_labels[0];

	for (int32_t i=1; i<m_train_labels.vlen; i++)
	{
		max_class=CMath::max(max_class, m_train_labels[i]);
		min_class=CMath::min(min_class, m_train_labels[i]);
	}

	for (int32_t i=0; i<m_train_labels.vlen; i++)
		m_train_labels[i]-=min_class;

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

	distance->precompute_lhs();

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

	distance->reset_precompute();

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

	switch (m_knn_solver)
	{
	case KNN_BRUTE:
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

		break;
	}
	case KNN_COVER_TREE: // Use cover tree
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

		break;
	}
	case KNN_KDTREE:
	{
		CFeatures* lhs = distance->get_lhs();
		CKDTree* kd_tree = new CKDTree(m_leaf_size);
		kd_tree->build_tree(dynamic_cast<CDenseFeatures<float64_t>*>(lhs));
		SG_UNREF(lhs);

		CFeatures* query = distance->get_rhs();
		kd_tree->query_knn(dynamic_cast<CDenseFeatures<float64_t>*>(query), m_k);
		SGMatrix<index_t> NN = kd_tree->get_knn_indices();
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
		SG_UNREF(query);
		break;
	}
	case KNN_LSH:
	{
		CDenseFeatures<float64_t>* features = dynamic_cast<CDenseFeatures<float64_t>*>(distance->get_lhs());
		std::vector<falconn::DenseVector<double>> feats;
		for(int32_t i=0; i < features->get_num_vectors(); i++)
		{
			int32_t len;
			bool free;
			float64_t* vec = features->get_feature_vector(i, len, free);
			falconn::DenseVector<double> temp = Map<VectorXd> (vec, len);
			feats.push_back(temp);
		}

		falconn::LSHConstructionParameters params 
			= falconn::get_default_parameters<falconn::DenseVector<double>>(features->get_num_vectors(),
                                   features->get_num_features(),
                                   falconn::DistanceFunction::EuclideanSquared,
                                   true);
		SG_UNREF(features);
		if (m_lsh_l && m_lsh_t)
			params.l = m_lsh_l;

		auto lsh_table = falconn::construct_table<falconn::DenseVector<double>>(feats, params);
		if (m_lsh_t)
			lsh_table->set_num_probes(m_lsh_t);

		CDenseFeatures<float64_t>* query_features = dynamic_cast<CDenseFeatures<float64_t>*>(distance->get_rhs());
		std::vector<falconn::DenseVector<double>> query_feats;

		SGMatrix<index_t> NN (m_k, query_features->get_num_vectors());
		for(int32_t i=0; i < query_features->get_num_vectors(); i++)
		{
			int32_t len;
			bool free;
			float64_t* vec = query_features->get_feature_vector(i, len, free);
			falconn::DenseVector<double> temp = Map<VectorXd> (vec, len);
			auto indices = new std::vector<int32_t> ();
			lsh_table->find_k_nearest_neighbors(temp, (int_fast64_t)m_k, indices);
			memcpy(NN.get_column_vector(i), indices->data(), sizeof(int32_t)*m_k);
			delete indices;
		}
		
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
		SG_UNREF(query_features);
		break;
	}
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

	distance->precompute_lhs();

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

	distance->reset_precompute();

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
	
	switch (m_knn_solver)
	{
	case KNN_COVER_TREE: // Use cover tree
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
		break;
	}
	case KNN_KDTREE:
	{
		//allocation for distances to nearest neighbors
		float64_t* dists=SG_MALLOC(float64_t, m_k);

		CFeatures* lhs = distance->get_lhs();
		CKDTree* kd_tree = new CKDTree(m_leaf_size);
		kd_tree->build_tree(dynamic_cast<CDenseFeatures<float64_t>*>(lhs));
		SG_UNREF(lhs);

		CFeatures* data = distance->get_rhs();
		kd_tree->query_knn(dynamic_cast<CDenseFeatures<float64_t>*>(data), m_k);
		SGMatrix<index_t> NN = kd_tree->get_knn_indices();
		for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
		{
			//write the labels of the k nearest neighbors from theirs indices
			for (int32_t j=0; j<m_k; j++)
			{
				train_lab[j] = m_train_labels[ NN(j,i) ];
				dists[j] = distance->distance(i, NN(j,i));
			}
			CMath::qsort_index(dists, train_lab, m_k);

			choose_class_for_multiple_k(output+i, classes, train_lab, num_lab);
		}
		break;
	}
	default:
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
