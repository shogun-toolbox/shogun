/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Sebastian Henschel
 * Copyright (C) 2008 Friedrich Miescher Laboratory of Max-Planck-Society
 */

#include "evaluation/PerformanceMeasures.h"

#include "lib/ShogunException.h"
#include "lib/Mathematics.h"

CPerformanceMeasures::CPerformanceMeasures()
: CSGObject(), true_labels(NULL), all_positives(0), all_negatives(0),
	sorted_true_labels(NULL), output(NULL), sorted_output(NULL)
{
}

CPerformanceMeasures::CPerformanceMeasures(
	CLabels* true_labels_, CLabels* output_)
: CSGObject(), all_positives(0), all_negatives(0),
	sorted_true_labels(NULL), sorted_output(NULL)
{
	init(true_labels_, output_);
}

CPerformanceMeasures::~CPerformanceMeasures()
{
	if (true_labels)
		SG_UNREF(true_labels);
	if (sorted_true_labels)
		delete sorted_true_labels;
	if (output)
		SG_UNREF(output);
	if (sorted_output)
		delete sorted_output;
}

void CPerformanceMeasures::init(
	CLabels* true_labels_, CLabels* output_)
{
	ASSERT(true_labels_);
	ASSERT(output_);
	INT num_labels=true_labels_->get_num_labels();
	if (num_labels!=output_->get_num_labels())
		throw ShogunException("Number of labels in true_labels and output differ!\n");

	for (INT i=0; i<num_labels; i++) {
		DREAL lab=true_labels_->get_label(i);
		if (lab==1.)
			all_positives++;
		else if (lab==-1.)
			all_negatives++;
		else
			throw ShogunException("Illegal true_labels, not {-1, 1}!\n");
	}

	true_labels=true_labels_;
	SG_REF(true_labels);
	output=output_;
	SG_REF(output);
}

void CPerformanceMeasures::ROC_sort()
{
	INT num_labels;

	if (sorted_output)
		delete sorted_output;
	sorted_output=output->get_labels(num_labels);

	if (sorted_true_labels)
		delete sorted_true_labels;
	sorted_true_labels=true_labels->get_labels(num_labels);

	CMath::qsort_backward_index(sorted_output, sorted_true_labels, num_labels);
}

void CPerformanceMeasures::compute_ROC(DREAL** result, INT* dim, INT *num)
{
	if (all_positives<1)
		throw ShogunException("Need at least one positive example in true_labels!\n");
	if (all_negatives<1)
		throw ShogunException("Need at least one negative example in true_labels!\n");

	INT num_labels=true_labels->get_num_labels();
	*dim=num_labels+1;
	*num=2;
	size_t sz=sizeof(DREAL)*(*dim)*(*num);
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for ROC result!\n");

	ROC_sort();

	DREAL false_positives=0.;
	DREAL true_positives=0.;
	INT i=0;
	DREAL prev=CMath::ALMOST_NEG_INFTY;

	for (i=0; i<num_labels; i++) {
		DREAL out=sorted_output[i];
		if (out!=prev) {
			(*result)[i]=false_positives/all_negatives;
			(*result)[*dim+i]=true_positives/all_positives;
			prev=out;
		}

		if (sorted_true_labels[i]==1)
			true_positives++;
		else
			false_positives++;
	}
	(*result)[i]=false_positives/all_negatives;
	(*result)[*dim+i]=true_positives/all_positives;

	/*
	delete sorted_output; sorted_output=NULL;
	delete sorted_true_labels; sorted_true_labels=NULL;
	*/
}
