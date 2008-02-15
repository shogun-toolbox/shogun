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
: CSGObject(), true_labels(NULL), output(NULL), sortedROC(NULL)
{
	try {
		init(NULL, NULL);
	} catch (ShogunException(e)) {}
}

CPerformanceMeasures::CPerformanceMeasures(
	CLabels* true_labels_, CLabels* output_)
: CSGObject(), true_labels(NULL), output(NULL), sortedROC(NULL)
{
	init(true_labels_, output_);
}

CPerformanceMeasures::~CPerformanceMeasures()
{
	if (true_labels) SG_UNREF(true_labels);
	if (output) SG_UNREF(output);
	if (sortedROC) free(sortedROC);
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::init(CLabels* true_labels_, CLabels* output_)
{
	all_true=0;
	all_false=0;
	auROC=0.;
	accuracy0=0.;
	auPRC=0.;
	fmeasure0=0.;

	if (!true_labels_)
		throw ShogunException("No true labels given!\n");
	if (!output_)
		throw ShogunException("No output given!\n");

	DREAL* labels=true_labels_->get_labels(num_labels);
	if (num_labels!=output_->get_num_labels()) {
		delete labels;
		throw ShogunException("Number of true labels and output labels differ!\n");
	}
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	if (sortedROC) {
		delete sortedROC;
		sortedROC=NULL;
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
			all_true++;
		else if (labels[i]==-1.)
			all_false++;
		else {
			delete labels;
			throw ShogunException("Illegal true labels, not purely {-1, 1}!\n");
		}
	}
	free(labels);

	true_labels=true_labels_;
	SG_REF(true_labels);
	output=output_;
	SG_REF(output);

	create_sortedROC();
}

void CPerformanceMeasures::create_sortedROC()
{
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(INT)*num_labels;
	if (sortedROC) delete sortedROC;
	sortedROC=(INT*) malloc(sz);
	if (!sortedROC)
		throw ShogunException("Could not allocate memory for sorted ROC index!\n");

	for (INT i=0; i<num_labels; i++) sortedROC[i]=i;
	DREAL* out=output->get_labels(num_labels);
	CMath::qsort_backward_index(out, sortedROC, num_labels);
	free(out);
}

/////////////////////////////////////////////////////////////////////

template <class T> DREAL CPerformanceMeasures::trapezoid_area(T x1, T x2, T y1, T y2)
{
	DREAL base=CMath::abs(x1-x2);
	DREAL height_avg=.5*(y1+y2);
	return base*height_avg;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_ROC(DREAL** result, INT *dim, INT *num)
{
	*dim=num_labels+1;
	*num=2;
	compute_ROC(result);
}

void CPerformanceMeasures::compute_ROC(DREAL** result)
{
	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (all_true<1)
		throw ShogunException("Need at least one positive example in true labels!\n");
	if (all_false<1)
		throw ShogunException("Need at least one negative example in true labels!\n");

	// num_labels+1 due to point 1,1
	INT num_roc=num_labels+1;
	size_t sz=sizeof(DREAL)*num_roc*2;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for ROC result!\n");

	INT fp=0;
	INT tp=0;
	INT fp_prev=0;
	INT tp_prev=0;
	DREAL out_prev=CMath::ALMOST_NEG_INFTY;
	auROC=0.;
	INT i;

	for (i=0; i<num_labels; i++) {
		DREAL out=output->get_label(sortedROC[i]);
		if (out!=out_prev) {
			(*result)[i]=(DREAL) fp/all_false;
			(*result)[num_roc+i]=(DREAL) tp/all_true;
			auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);

			fp_prev=fp;
			tp_prev=tp;
			out_prev=out;
		}

		if (true_labels->get_label(sortedROC[i])==1) {
			tp++;
		} else {
			fp++;
		}
	}

	// calculate for 1,1
	(*result)[i]=(DREAL) fp/all_false;
	(*result)[num_roc+i]=(DREAL) tp/all_true;

	/* paper says:
	 * auROC+=trapezoid_area(1, fp_prev, 1, tp_prev)
	 * wrong? was meant for calculating with rates?
	 */
	auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);
	auROC/=all_true*all_false; // normalise
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_accuracy0()
{
	if (accuracy0!=0.) return accuracy0;

	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	INT tp=0;
	INT tn=0;

	for (INT i=0; i<num_labels; i++) {
		INT out=CMath::sign(output->get_label(i));
		if (out>0 && true_labels->get_label(i)>0) tp++;
		else if (out<0 && true_labels->get_label(i)<0) tn++;
	}
	accuracy0=(DREAL) (tp+tn)/num_labels;

	return accuracy0;
}

void CPerformanceMeasures::compute_accuracyROC(DREAL** result, bool do_error)
{
	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for accuracy!\n");

	DREAL out_prev=CMath::ALMOST_NEG_INFTY;
	INT tp=0;
	INT tn=all_false;

	for (INT i=0; i<num_labels; i++) {
		DREAL out=output->get_label(sortedROC[i]);
		if (out!=out_prev) {
			(*result)[i]=(DREAL) (tp+tn)/num_labels;
			if (do_error) (*result)[i]=1.-(*result)[i];
			out_prev=out;
		}

		if (true_labels->get_label(sortedROC[i])==1) {
			tp++;
		} else {
			tn--;
		}
	}
}

void CPerformanceMeasures::get_accuracyROC(DREAL** result, INT* num)
{
	*num=num_labels;
	compute_accuracyROC(result, false);
}

void CPerformanceMeasures::get_errorROC(DREAL** result, INT *num)
{
	*num=num_labels;
	compute_accuracyROC(result, true);
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_PRC(DREAL** result, INT *dim, INT *num)
{
	*dim=num_labels;
	*num=2;
	compute_PRC(result);
}

// FIXME: make as efficient as compute_ROC
void CPerformanceMeasures::compute_PRC(DREAL** result)
{
	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*num_labels*2;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for PRC result!\n");

	INT fp=0;
	INT tp=0;

	for (INT i=0; i<num_labels; i++) {
		tp=0;
		fp=0;
		DREAL threshold=output->get_label(i);

		for (INT j=0; j<num_labels; j++) {
			if (output->get_label(j)>=threshold) {
				if (true_labels->get_label(j)>0) tp++;
				else fp++;
			}
		}

		(*result)[i]=(DREAL) tp/all_true; // recall
		(*result)[num_labels+i]=(DREAL) tp/(tp+fp); // precision
	}

	// sort by ascending recall
	CMath::qsort_index(*result, (*result)+num_labels, num_labels);

	// calculate auPRC
	auPRC=0.;
	for (INT i=0; i<num_labels-1; i++) {
		if ((*result)[1+i]==(*result)[i]) continue;
		auPRC+=trapezoid_area((*result)[1+i], (*result)[i], (*result)[1+num_labels+i], (*result)[num_labels+i]);
	}
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_fmeasurePRC(DREAL** result, INT* num)
{
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for F-measure!\n");

	*num=num_labels;
	DREAL** prc;
	compute_PRC(prc);

	for (INT i=0; i<num_labels; i++) {
		(*result)[i]=2./(1./(*prc)[i+num_labels]+1./(*prc)[i]);
	}
	free(*prc);
}

DREAL CPerformanceMeasures::get_fmeasure0()
{
	if (fmeasure0!=0.) return fmeasure0;

	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	INT tp=0;
	INT fp=0;
	DREAL precision;
	DREAL recall;

	for (INT i=0; i<num_labels; i++) {
		if (CMath::sign(output->get_label(i))>0) {
			if (true_labels->get_label(i)>0) tp++;
			else fp++;
		}
	}

	recall=(DREAL) tp/all_true;
	precision=(DREAL) tp/(tp+fp);
	fmeasure0=2./(1./precision+1./recall);

	return fmeasure0;
}

