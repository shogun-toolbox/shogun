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

#include <shogun/base/Parameter.h>
#include <shogun/base/progress.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/KNN.h>

//#define DEBUG_KNN

using namespace shogun;

CKNN::CKNN()
: CDistanceMachine()
{
	init();
}

CKNN::CKNN(index_t k, CDistance* d, CLabels* trainlab, KNN_SOLVER knn_solver)
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
	solver=NULL;
#ifdef HAVE_CXX11
	m_lsh_l = 0;
	m_lsh_t = 0;
#endif

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

	SGVector<index_t> lab = ((CMulticlassLabels*)m_labels)->get_int_labels();
	m_train_labels=lab.clone();
	ASSERT(m_train_labels.vlen>0)

	index_t max_class = m_train_labels[0];
	index_t min_class = m_train_labels[0];

	for (index_t i = 1; i < m_train_labels.vlen; i++)
	{
		max_class=CMath::max(max_class, m_train_labels[i]);
		min_class=CMath::min(min_class, m_train_labels[i]);
	}

	for (index_t i = 0; i < m_train_labels.vlen; i++)
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
	index_t n = distance->get_num_vec_rhs();
	//distances to train data
	SGVector<float64_t> dists(m_train_labels.vlen);
	//indices to train data
	SGVector<index_t> train_idxs(m_train_labels.vlen);
	//pre-allocation of the nearest neighbors
	SGMatrix<index_t> NN(m_k, n);

	distance->precompute_lhs();
	distance->precompute_rhs();

	auto pb = progress(range(n), *this->io);

	//for each test example
	for (index_t i = 0; i < n && (!cancel_computation()); i++)
	{
		pb.print_progress();

		//lhs idx 0..num train examples-1 (i.e., all train examples) and rhs idx i
		distances_lhs(dists,0,m_train_labels.vlen-1,i);

		//fill in an array with 0..num train examples-1
		for (index_t j = 0; j < m_train_labels.vlen; j++)
			train_idxs[j]=j;

		//sort the distance vector between test example i and all train examples
		CMath::qsort_index(dists.vector, train_idxs.vector, m_train_labels.vlen);

#ifdef DEBUG_KNN
		SG_PRINT("\nQuick sort query %d\n", i)
		for (index_t j = 0; j < m_k; j++)
			SG_PRINT("%d ", train_idxs[j])
		SG_PRINT("\n")
#endif

		//fill in the output the indices of the nearest neighbors
		for (index_t j = 0; j < m_k; j++)
			NN(j,i) = train_idxs[j];
	}
	pb.complete();

	distance->reset_precompute();

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

	index_t num_lab = distance->get_num_vec_rhs();
	ASSERT(m_k<=distance->get_num_vec_lhs())

	//labels of the k nearest neighbors
	SGVector<index_t> train_lab(m_k);

	SG_INFO("%d test examples\n", num_lab)

	//histogram of classes and returned output
	SGVector<float64_t> classes(m_num_classes);

	init_solver(m_knn_solver);

	CMulticlassLabels* output = solver->classify_objects(distance, num_lab, train_lab, classes);

	SG_UNREF(solver);

	return output;
}

CMulticlassLabels* CKNN::classify_NN()
{
	ASSERT(distance)
	ASSERT(m_num_classes>0)

	index_t num_lab = distance->get_num_vec_rhs();
	ASSERT(num_lab)

	CMulticlassLabels* output = new CMulticlassLabels(num_lab);
	SGVector<float64_t> distances(m_train_labels.vlen);

	SG_INFO("%d test examples\n", num_lab)

	distance->precompute_lhs();

	auto pb = progress(range(num_lab), *this->io);

	// for each test example
	for (index_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		pb.print_progress();

		// get distances from i-th test example to 0..num_m_train_labels-1 train examples
		distances_lhs(distances,0,m_train_labels.vlen-1,i);
		index_t j;

		// assuming 0th train examples as nearest to i-th test example
		index_t out_idx = 0;
		float64_t min_dist = distances.vector[0];

		// searching for nearest neighbor by comparing distances
		for (j=0; j<m_train_labels.vlen; j++)
		{
			if (distances.vector[j]<min_dist)
			{
				min_dist = distances.vector[j];
				out_idx = j;
			}
		}

		// label i-th test example with label of nearest neighbor with out_idx index
		output->set_label(i,m_train_labels.vector[out_idx]+m_min_label);
	}
	pb.complete();

	distance->reset_precompute();

	return output;
}

SGMatrix<index_t> CKNN::classify_for_multiple_k()
{
	ASSERT(m_num_classes>0)
	ASSERT(distance)
	ASSERT(distance->get_num_vec_rhs())

	index_t num_lab = distance->get_num_vec_rhs();
	ASSERT(m_k<=num_lab)

	//working buffer of m_train_labels
	SGVector<index_t> train_lab(m_k);

	//histogram of classes and returned output
	SGVector<index_t> classes(m_num_classes);

	SG_INFO("%d test examples\n", num_lab)

	init_solver(m_knn_solver);

	SGVector<index_t> output =
	    solver->classify_objects_k(distance, num_lab, train_lab, classes);

	SG_UNREF(solver);

	return SGMatrix<index_t>(output, num_lab, m_k);
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

void CKNN::init_solver(KNN_SOLVER knn_solver)
{
	switch (knn_solver)
	{
	case KNN_BRUTE:
	{
		SGMatrix<index_t> NN = nearest_neighbors();
		solver = new CBruteKNNSolver(m_k, m_q, m_num_classes, m_min_label, m_train_labels, NN);
		SG_REF(solver);
		break;
	}
	case KNN_KDTREE:
	{
		solver = new CKDTREEKNNSolver(m_k, m_q, m_num_classes, m_min_label, m_train_labels, m_leaf_size);
		SG_REF(solver);
		break;
	}
	case KNN_COVER_TREE:
	{
		solver = new CCoverTreeKNNSolver(m_k, m_q, m_num_classes, m_min_label, m_train_labels);
		SG_REF(solver);
		break;
	}
#ifdef HAVE_CXX11
	case KNN_LSH:
	{
		solver = new CLSHKNNSolver(m_k, m_q, m_num_classes, m_min_label, m_train_labels, m_lsh_l, m_lsh_t);
		SG_REF(solver);
		break;
	}
#endif
	}
}
