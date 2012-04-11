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

#include <shogun/classifier/KNN.h>
#include <shogun/features/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parameter.h>

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

	ASSERT(d);
	ASSERT(trainlab);

	set_distance(d);
	set_labels(trainlab);
	train_labels.vlen=trainlab->get_num_labels();
}

void CKNN::init()
{
	/* do not store model features by default (CDistanceMachine::apply(...) is
	 * overwritten */
	set_store_model_features(false);

	m_k=3;
	m_q=1.0;
	m_use_covertree=false;
	num_classes=0;
	m_covertree=NULL;
	m_built_covertree=false;

	/** TODO not really sure here if these two first guys should be MS_AVAILABLE or 
	 *  MS_NOT_AVAILABLE
	 */
	SG_ADD(&m_k, "m_k", "Parameter k", MS_AVAILABLE);
	SG_ADD(&m_q, "m_q", "Parameter q", MS_AVAILABLE);
	//SG_ADD(&m_use_covertree, "m_use_covertree", "Parameter use_covertree", MS_NOT_AVAILABLE);
	SG_ADD(&m_built_covertree, "m_built_covertree", "Parameter built_covertree", MS_NOT_AVAILABLE);
	SG_ADD(&num_classes, "num_classes", "Number of classes", MS_NOT_AVAILABLE);
	//SG_ADD((CSGObject**) &m_covertree, "m_covertree", "Member cover tree", MS_NOT_AVAILABLE);
}

CKNN::~CKNN()
{
	SG_FREE(train_labels.vector);
	if ( m_use_covertree )
		delete m_covertree;
}

bool CKNN::train_machine(CFeatures* data)
{
	ASSERT(m_labels);
	ASSERT(distance);

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		distance->init(data, data);
	}

	SGVector<int32_t> lab=m_labels->get_int_labels();
	train_labels.vlen=lab.vlen;
	train_labels.vector=CMath::clone_vector(lab.vector, lab.vlen);
	lab.free_vector();
	ASSERT(train_labels.vlen>0);

	int32_t max_class=train_labels.vector[0];
	int32_t min_class=train_labels.vector[0];

	for (int32_t i=1; i<train_labels.vlen; i++)
	{
		max_class=CMath::max(max_class, train_labels.vector[i]);
		min_class=CMath::min(min_class, train_labels.vector[i]);
	}

	for (int32_t i=0; i<train_labels.vlen; i++)
		train_labels.vector[i]-=min_class;

	min_label=min_class;
	num_classes=max_class-min_class+1;

	SG_INFO( "num_classes: %d (%+d to %+d) num_train: %d\n", num_classes,
			min_class, max_class, train_labels.vlen);

	// If cover tree is to be used, populate it with training vectors
	// assuming that distance(train_vectors, train_vectors)
	if ( m_use_covertree )
	{
		// Ensure that distance has the same features lhs and rhs
		if ( ! distance->lhs_equals_rhs() )
		{
			SG_ERROR("Features lhs and rhs must be equal to train KNN "
				 "with covertree support\n");
		}

		// Look for the max distance among training vectors
		float64_t max_dist = 0.0;
		for (int32_t i=0; i<train_labels.vlen; i++)
		{
			for (int32_t j=i+1; j<train_labels.vlen; j++)
				max_dist = CMath::max(max_dist, distance->distance(i, j));
		}

		// Create cover tree
		m_covertree = new CoverTree<KNN_COVERTREE_POINT>(max_dist);

		// Insert training vectors
		for (int32_t i=0; i<train_labels.vlen; i++)
			m_covertree->insert(KNN_COVERTREE_POINT(i, distance));

		m_built_covertree = true;
	}
	else
	{
		m_built_covertree = false;
	}

	return true;
}

CLabels* CKNN::apply()
{
	ASSERT(num_classes>0);
	ASSERT(distance);
	ASSERT(distance->get_num_vec_rhs());

	if ( m_use_covertree && ! m_built_covertree )
		SG_ERROR("The cover tree must have been built during training to use "
			 "it for classification\n");

	int32_t num_lab=distance->get_num_vec_rhs();
	ASSERT(m_k<=distance->get_num_vec_lhs());

	CLabels* output=new CLabels(num_lab);

	float64_t* dists   = NULL;
	int32_t* train_lab = NULL;
	//vector of neighbors used for the cover tree support
	int32_t* nearest_neighbors = NULL;
	//distances to train data and working buffer of train_labels
	if ( ! m_use_covertree )
	{
		dists=SG_MALLOC(float64_t, train_labels.vlen);
		train_lab=SG_MALLOC(int32_t, train_labels.vlen);
	}
	else
	{
		train_lab=SG_MALLOC(int32_t, m_k);
		nearest_neighbors=SG_MALLOC(int32_t, m_k);
	}

	SG_INFO( "%d test examples\n", num_lab);
	CSignal::clear_cancel();

	///histogram of classes and returned output
	float64_t* classes=SG_MALLOC(float64_t, num_classes);

	for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		SG_PROGRESS(i, 0, num_lab);

		if ( ! m_use_covertree )
		{
			//lhs idx 1..n and rhs idx i
			distances_lhs(dists,0,train_labels.vlen-1,i);

			for (int32_t j=0; j<train_labels.vlen; j++)
				train_lab[j]=train_labels.vector[j];

			//sort the distance vector for test example j to all train examples
			//classes[1..k] then holds the classes for minimum distance
			CMath::qsort_index(dists, train_lab, train_labels.vlen);
		}
		else
		{
			//get the k nearest neighbors to test vector i using the cover tree 
			get_neighbors(nearest_neighbors, i);

			//translate from indices to labels of the nearest neighbors
			for (int32_t j=0; j<m_k; j++)
				train_lab[j]=train_labels.vector[ nearest_neighbors[j] ];
		}

		//compute histogram of class outputs of the first k nearest neighbours
		for (int32_t j=0; j<num_classes; j++)
			classes[j]=0.0;

		float64_t multiplier = m_q;
		for (int32_t j=0; j<m_k; j++)
		{
			classes[train_lab[j]]+= multiplier;
			multiplier*= multiplier;
		}

		//choose the class that got 'outputted' most often
		int32_t out_idx=0;
		float64_t out_max=0;

		for (int32_t j=0; j<num_classes; j++)
		{
			if (out_max< classes[j])
			{
				out_idx= j;
				out_max= classes[j];
			}
		}
		output->set_label(i, out_idx+min_label);
	}

	SG_FREE(classes);
	SG_FREE(train_lab);
	if ( ! m_use_covertree )
		SG_FREE(dists);
	else
		SG_FREE(nearest_neighbors);

	return output;
}

CLabels* CKNN::apply(CFeatures* data)
{
	init_distance(data);

	// redirecting to fast (without sorting) classify if k==1
	if (m_k == 1)
		return classify_NN();

	return apply();
}

CLabels* CKNN::classify_NN()
{
	ASSERT(distance);
	ASSERT(num_classes>0);

	int32_t num_lab = distance->get_num_vec_rhs();
	ASSERT(num_lab);

	CLabels* output = new CLabels(num_lab);
	float64_t* distances = SG_MALLOC(float64_t, train_labels.vlen);

	SG_INFO("%d test examples\n", num_lab);
	CSignal::clear_cancel();

	// for each test example
	for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		SG_PROGRESS(i,0,num_lab);

		// get distances from i-th test example to 0..num_train_labels-1 train examples
		distances_lhs(distances,0,train_labels.vlen-1,i);
		int32_t j;

		// assuming 0th train examples as nearest to i-th test example
		int32_t out_idx = 0;
		float64_t min_dist = distances[0];

		// searching for nearest neighbor by comparing distances
		for (j=0; j<train_labels.vlen; j++)
		{
			if (distances[j]<min_dist)
			{
				min_dist = distances[j];
				out_idx = j;
			}
		}

		// label i-th test example with label of nearest neighbor with out_idx index
		output->set_label(i,train_labels.vector[out_idx]+min_label);
	}

	delete [] distances;
	return output;
}

SGMatrix<int32_t> CKNN::classify_for_multiple_k()
{
	ASSERT(num_classes>0);
	ASSERT(distance);
	ASSERT(distance->get_num_vec_rhs());

	if ( m_use_covertree && ! m_built_covertree )
		SG_ERROR("The cover tree must have been built during training to use "
			 "it for classification\n");

	int32_t num_lab=distance->get_num_vec_rhs();
	ASSERT(m_k<=num_lab);

	int32_t* output=SG_MALLOC(int32_t, m_k*num_lab);

	float64_t* dists;
	int32_t* train_lab;
	//vector of neighbors used for the cover tree support
	int32_t* nearest_neighbors;
	//distances to train data and working buffer of train_labels
	if ( ! m_use_covertree )
	{
		dists=SG_MALLOC(float64_t, train_labels.vlen);
		train_lab=SG_MALLOC(int32_t, train_labels.vlen);
	}
	else
	{
		train_lab=SG_MALLOC(int32_t, m_k);
		nearest_neighbors=SG_MALLOC(int32_t, m_k);
	}

	///histogram of classes and returned output
	int32_t* classes=SG_MALLOC(int32_t, num_classes);

	SG_INFO( "%d test examples\n", num_lab);
	CSignal::clear_cancel();

	for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		SG_PROGRESS(i, 0, num_lab);

		if ( ! m_use_covertree )
		{
			// lhs idx 1..n and rhs idx i
			distances_lhs(dists,0,train_labels.vlen-1,i);
			for (int32_t j=0; j<train_labels.vlen; j++)
				train_lab[j]=train_labels.vector[j];

			//sort the distance vector for test example j to all train examples
			//classes[1..k] then holds the classes for minimum distance
			CMath::qsort_index(dists, train_lab, train_labels.vlen);
		}
		else
		{
			//get the k nearest neighbors to test vector i using the cover tree 
			get_neighbors(nearest_neighbors, i);

			//translate from indices to labels of the nearest neighbors
			for (int32_t j=0; j<m_k; j++)
				train_lab[j]=train_labels.vector[ nearest_neighbors[j] ];
		}

		//compute histogram of class outputs of the first k nearest neighbours
		for (int32_t j=0; j<num_classes; j++)
			classes[j]=0;

		for (int32_t j=0; j<m_k; j++)
		{
			classes[train_lab[j]]++;

			//choose the class that got 'outputted' most often
			int32_t out_idx=0;
			int32_t out_max=0;

			for (int32_t c=0; c<num_classes; c++)
			{
				if (out_max< classes[c])
				{
					out_idx= c;
					out_max= classes[c];
				}
			}
			output[j*num_lab+i]=out_idx+min_label;
		}
	}

	SG_FREE(train_lab);
	SG_FREE(classes);
	if ( ! m_use_covertree )
		SG_FREE(dists);
	else
		SG_FREE(nearest_neighbors);

	return SGMatrix<int32_t>(output,num_lab,m_k,true);
}

void CKNN::init_distance(CFeatures* data)
{
	if (!distance)
		SG_ERROR("No distance assigned!\n");
	CFeatures* lhs=distance->get_lhs();
	if (!lhs || !lhs->get_num_vectors())
	{
		SG_UNREF(lhs);
		SG_ERROR("No vectors on left hand side\n");
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

void CKNN::get_neighbors(int32_t* out, int32_t idx)
{
	std::vector<KNN_COVERTREE_POINT> neighbors =
		m_covertree->kNearestNeighbors(KNN_COVERTREE_POINT(idx, distance), m_k);

	for (std::size_t m=0; m<unsigned(m_k); m++)
		out[m] = neighbors[m].m_point_index;
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
