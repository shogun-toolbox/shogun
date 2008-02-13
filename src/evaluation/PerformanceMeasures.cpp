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
: CSGObject(), true_labels(NULL), output(NULL), roc(NULL), accROC(NULL)
{
	try {
		init(NULL, NULL);
	} catch (ShogunException(e)) {}
}

CPerformanceMeasures::CPerformanceMeasures(
	CLabels* true_labels_, CLabels* output_)
: CSGObject(), true_labels(NULL), output(NULL), roc(NULL), accROC(NULL)
{
	init(true_labels_, output_);
}

CPerformanceMeasures::~CPerformanceMeasures()
{
	if (true_labels) SG_UNREF(true_labels);
	if (output) SG_UNREF(output);
	if (roc) delete roc;
	if (accROC) delete accROC;
}

void CPerformanceMeasures::init(CLabels* true_labels_, CLabels* output_)
{
	all_positives=0;
	all_negatives=0;
	auROC=0;
	accROC=0;

	if (!true_labels_)
		throw ShogunException("No true labels given!\n");
	if (!output_)
		throw ShogunException("No output given!\n");

	DREAL* labels=true_labels_->get_labels(num_labels);
	if (num_labels!=output_->get_num_labels()) {
		delete labels;
		throw ShogunException("Number of true labels and output labels differ!\n");
	}

	if (roc) {
		delete roc;
		roc=NULL;
	}
	if (accROC) {
		delete accROC;
		accROC=NULL;
	}
	if (true_labels) {
		SG_UNREF(true_labels);
		true_labels=NULL;
	}
	if (output) {
		SG_UNREF(output);
		output=NULL;
	}

	for (INT i=0; i<num_labels; i++) {
		if (labels[i]==1.)
			all_positives++;
		else if (labels[i]==-1.)
			all_negatives++;
		else {
			delete labels;
			throw ShogunException("Illegal true labels, not purely {-1, 1}!\n");
		}
	}
	delete labels;

	true_labels=true_labels_;
	SG_REF(true_labels);
	output=output_;
	SG_REF(output);
}


DREAL CPerformanceMeasures::trapezoid_area(INT x1, INT x2, INT y1, INT y2)
{
	INT base=CMath::abs(x1-x2);
	DREAL height_avg=(y1+y2)/2.;
	return base*height_avg;
}

void CPerformanceMeasures::compute_ROC()
{
	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (all_positives<1)
		throw ShogunException("Need at least one positive example in true_labels!\n");
	if (all_negatives<1)
		throw ShogunException("Need at least one negative example in true_labels!\n");

	// num_labels+1 due to point 1,1
	INT num_roc=num_labels+1;
	if (roc) delete roc;
	size_t sz=sizeof(DREAL)*num_roc*2;
	roc=new DREAL[sz];
	if (!roc)
		throw ShogunException("Could not allocate memory for ROC result!\n");

	if (accROC) delete accROC;
	sz=sizeof(DREAL)*num_labels;
	accROC=new DREAL[sz];
	if (!accROC)
		throw ShogunException("Could not allocate memory for accROC!\n");

	// sorting
	DREAL* sorted_output=output->get_labels(num_labels);
	DREAL* sorted_true_labels=true_labels->get_labels(num_labels);
	CMath::qsort_backward_index(sorted_output, sorted_true_labels, num_labels);

	// various states
	INT fp=0;
	INT fn=all_positives;
	INT tp=0;
	INT tn=all_negatives;
	INT fp_prev=0;
	INT tp_prev=0;
	DREAL out_prev=CMath::ALMOST_NEG_INFTY;

	// area under ROC
	auROC=0;

	INT i;
	for (i=0; i<num_labels; i++) {
		DREAL out=sorted_output[i];
		if (out!=out_prev) {
			roc[i]=(DREAL) fp/all_negatives;
			roc[num_roc+i]=(DREAL) tp/all_positives;
			auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);
			accROC[i]=(DREAL) (tp+tn)/(all_positives+all_negatives);

			fp_prev=fp;
			tp_prev=tp;
			out_prev=out;
		}

		if (sorted_true_labels[i]==1) {
			tp++;
			fn--; // fn + tp == all positives
		} else {
			fp++;
			tn--; // tn + fp == all negatives
		}
	}

	// calculate for 1,1
	roc[i]=(DREAL) fp/all_negatives;
	roc[num_roc+i]=(DREAL) tp/all_positives;
	/* paper says:
	 * auROC+=trapezoid_area(1, fp_prev, 1, tp_prev)
	 * wrong? was meant for calculating with rates?
	 */
	auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);
	// normalise: geometric means
	auROC/=all_positives*all_negatives;

	delete sorted_true_labels;
	delete sorted_output;
}

void CPerformanceMeasures::get_ROC(DREAL** result, INT *dim, INT *num)
{
	if (!roc) compute_ROC();

	*dim=num_labels+1;
	*num=2;
	size_t sz=sizeof(DREAL)*(*dim)*(*num);
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for ROC result!\n");

	memcpy(*result, roc, sz);
}

void CPerformanceMeasures::get_accROC(DREAL** result, INT *num)
{
	if (!accROC) compute_ROC();

	*num=num_labels;
	size_t sz=sizeof(DREAL)*num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for accuracy!\n");

	memcpy(*result, accROC, sz);
}

void CPerformanceMeasures::get_errROC(DREAL** result, INT *num)
{
	if (!accROC) compute_ROC();

	*num=num_labels;
	size_t sz=sizeof(DREAL)*num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for error rate!\n");

	memcpy(*result, accROC, sz);

	for (INT i=0; i<num_labels; i++) (*result)[i]=1.-(*result)[i];
}

/*
void CPerformanceMeasures::compute_accuracy()
{
	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (all_positives+all_negatives<1)
		throw ShogunException("The number of positive + negative true labels is less than 1 (which should never happen)!\n");

	INT tp=0;
	INT tn=0;
	DREAL* labels=true_labels->get_labels(num_labels);
	DREAL* out=output->get_labels(num_labels);

	size_t sz=sizeof(DREAL)*num_labels;
	if (accuracy) delete accuracy;
	accuracy=new DREAL[sz];
	if (!accuracy)
		throw ShogunException("Could not allocate memory for accuracy!\n");

	for (INT i=0; i<num_labels; i++) {
		DREAL threshold=out[i];
		for (INT j=0; j<num_labels; j++) {
			if (out[j]>=threshold && labels[j]==1) {
				tp++;
			} else if (out[j]<threshold && labels[j]==-1) {
				tn++;
			}
		}
		accuracy[i]=(DREAL) (tp+tn)/(all_positives+all_negatives);
		tp=tn=0;
	}

	delete labels;
	delete out;
}
*/

