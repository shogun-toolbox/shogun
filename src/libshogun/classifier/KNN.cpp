/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/KNN.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/Signal.h"

using namespace shogun;

CKNN::CKNN()
: CDistanceMachine(), k(3), num_classes(0), num_train_labels(0), train_labels(NULL)
{
}

CKNN::CKNN(int32_t k_, CDistance* d, CLabels* trainlab)
: CDistanceMachine(), k(k_), num_classes(0), train_labels(NULL)
{
	ASSERT(d);
	ASSERT(trainlab);

    set_distance(d);
    set_labels(trainlab);
    num_train_labels=trainlab->get_num_labels();
}


CKNN::~CKNN()
{
	delete[] train_labels;
}

bool CKNN::train(CFeatures* data)
{
	ASSERT(labels);
	ASSERT(distance);

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		distance->init(data, data);
	}

	train_labels=labels->get_int_labels(num_train_labels);
	ASSERT(train_labels);
	ASSERT(num_train_labels>0);

	int32_t max_class=train_labels[0];
	int32_t min_class=train_labels[0];

	int32_t i;
	for (i=1; i<num_train_labels; i++)
	{
		max_class=CMath::max(max_class, train_labels[i]);
		min_class=CMath::min(min_class, train_labels[i]);
	}

	for (i=0; i<num_train_labels; i++)
		train_labels[i]-=min_class;

	min_label=min_class;
	num_classes=max_class-min_class+1;

	SG_INFO( "num_classes: %d (%+d to %+d) num_train: %d\n", num_classes, min_class, max_class, num_train_labels);
	return true;
}

CLabels* CKNN::classify()
{
	ASSERT(num_classes>0);
	ASSERT(distance);
	ASSERT(distance->get_num_vec_rhs());

	int32_t num_lab=distance->get_num_vec_rhs();
	ASSERT(k<=num_lab);

	CLabels* output=new CLabels(num_lab);

	//distances to train data and working buffer of train_labels
	float64_t* dists=new float64_t[num_train_labels];
	int32_t* train_lab=new int32_t[num_train_labels];

	///histogram of classes and returned output
	int32_t* classes=new int32_t[num_classes];

	ASSERT(dists);
	ASSERT(train_lab);
	ASSERT(classes);

	SG_INFO( "%d test examples\n", num_lab);
	CSignal::clear_cancel();

	for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		SG_PROGRESS(i, 0, num_lab);

		// lhs idx 1..n and rhs idx i
		distances_lhs(dists,0,num_train_labels-1,i);
		int32_t j;
		for (j=0; j<num_train_labels; j++)
			train_lab[j]=train_labels[j];

		//sort the distance vector for test example j to all train examples
		//classes[1..k] then holds the classes for minimum distance
		CMath::qsort_index(dists, train_lab, num_train_labels);

		//compute histogram of class outputs of the first k nearest neighbours
		for (j=0; j<num_classes; j++)
			classes[j]=0;

		for (j=0; j<k; j++)
			classes[train_lab[j]]++;

		//choose the class that got 'outputted' most often
		int32_t out_idx=0;
		int32_t out_max=0;

		for (j=0; j<num_classes; j++)
		{
			if (out_max< classes[j])
			{
				out_idx= j;
				out_max= classes[j];
			}
		}

		output->set_label(i, out_idx+min_label);
	}

	delete[] dists;
	delete[] train_lab;
	delete[] classes;

	return output;
}

CLabels* CKNN::classify(CFeatures* data)
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

	return classify();
}

void CKNN::classify_for_multiple_k(int32_t** dst, int32_t* num_vec, int32_t* k_out)
{
	ASSERT(dst);
	ASSERT(k_out);
	ASSERT(num_vec);

	ASSERT(num_classes>0);
	ASSERT(distance);
	ASSERT(distance->get_num_vec_rhs());

	int32_t num_lab=distance->get_num_vec_rhs();
	ASSERT(k<=num_lab);

	int32_t* output=(int32_t*) malloc(sizeof(int32_t)*k*num_lab);

	//distances to train data and working buffer of train_labels
	float64_t* dists=new float64_t[num_train_labels];
	int32_t* train_lab=new int32_t[num_train_labels];

	///histogram of classes and returned output
	int32_t* classes=new int32_t[num_classes];

	SG_INFO( "%d test examples\n", num_lab);
	CSignal::clear_cancel();

	for (int32_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		SG_PROGRESS(i, 0, num_lab);

		// lhs idx 1..n and rhs idx i
		distances_lhs(dists,0,num_train_labels-1,i);
		for (int32_t j=0; j<num_train_labels; j++)
			train_lab[j]=train_labels[j];

		//sort the distance vector for test example j to all train examples
		//classes[1..k] then holds the classes for minimum distance
		CMath::qsort_index(dists, train_lab, num_train_labels);

		//compute histogram of class outputs of the first k nearest neighbours
		for (int32_t j=0; j<num_classes; j++)
			classes[j]=0;

		for (int32_t j=0; j<k; j++)
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

	delete[] dists;
	delete[] train_lab;
	delete[] classes;

	*dst=output;
	*k_out=k;
	*num_vec=num_lab;
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
