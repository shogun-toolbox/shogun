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
	if (sortedROC) delete[] sortedROC;
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
	cc0=0.;
	wr_acc0=0.;
	balance0=0.;

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
	delete[] labels;

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
	if (sortedROC) delete[] sortedROC;
	sortedROC=new INT[sz];
	if (!sortedROC)
		throw ShogunException("Could not allocate memory for sorted ROC index!\n");

	for (INT i=0; i<num_labels; i++) sortedROC[i]=i;
	DREAL* out=output->get_labels(num_labels);
	CMath::qsort_backward_index(out, sortedROC, num_labels);
	delete[] out;
}

/////////////////////////////////////////////////////////////////////

template <class T> DREAL CPerformanceMeasures::trapezoid_area(T x1, T x2, T y1, T y2)
{
	DREAL base=CMath::abs(x1-x2);
	DREAL height_avg=.5*(y1+y2);
	return base*height_avg;
}

INT* CPerformanceMeasures::check_classification(DREAL threshold)
{
	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	INT num_results=4;
	INT* results=new INT[num_results];
	for (INT i=0; i<num_results; i++) results[i]=0;

	for (INT i=0; i<num_labels; i++) {
		if (output->get_label(i)>=threshold) {
			if (true_labels->get_label(i)>0) results[0]++;
			else results[1]++;
		} else {
			if (true_labels->get_label(i)>0) results[2]++;
			else results[3]++;
		}
	}

	return results;
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

	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT tn=checked[3];
	accuracy0=(DREAL) (tp+tn)/num_labels;

	delete[] checked;
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
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*num_labels*2;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for PRC result!\n");

	INT* checked;
	INT tp;
	INT fp;
	DREAL threshold;

	for (INT i=0; i<num_labels; i++) {
		threshold=output->get_label(i);
		checked=check_classification(threshold);
		tp=checked[0];
		fp=checked[1];
		(*result)[i]=(DREAL) tp/all_true; // recall
		(*result)[num_labels+i]=(DREAL) tp/(tp+fp); // precision

		delete[] checked;
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

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT fp=checked[1];
	DREAL recall=(DREAL) tp/all_true;
	DREAL precision=(DREAL) tp/(tp+fp);
	fmeasure0=2./(1./precision+1./recall);

	delete[] checked;
	return fmeasure0;
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_CC0()
{
	if (cc0!=0.) return cc0;

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT fp=checked[1];
	INT fn=checked[2];
	INT tn=checked[3];
	cc0=(tp*tn-fp*fn)/CMath::sqrt((DREAL) (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

	delete[] checked;
	return cc0;
}

void CPerformanceMeasures::get_CC(DREAL** result, INT* num)
{
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	*num=num_labels;
	size_t sz=sizeof(DREAL)*num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for CC!\n");

	INT* checked;
	INT tp;
	INT fp;
	INT fn;
	INT tn;
	DREAL threshold;

	for (INT i=0; i<num_labels; i++) {
		threshold=output->get_label(i);
		checked=check_classification(threshold);
		tp=checked[0];
		fp=checked[1];
		fn=checked[2];
		tn=checked[3];
		(*result)[i]=(tp*tn-fp*fn)/CMath::sqrt((DREAL) (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
		delete[] checked;
	}
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_WRacc0()
{
	if (wr_acc0!=0.) return wr_acc0;

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT fp=checked[1];
	INT fn=checked[2];
	INT tn=checked[3];
	wr_acc0=(DREAL) tp/(tp+fn)-fp/(fp+tn);

	delete[] checked;
	return wr_acc0;
}

void CPerformanceMeasures::get_WRacc(DREAL** result, INT* num)
{
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	*num=num_labels;
	size_t sz=sizeof(DREAL)*num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for WR accuracy!\n");

	INT* checked;
	INT tp;
	INT fp;
	INT fn;
	INT tn;
	DREAL threshold;

	for (INT i=0; i<num_labels; i++) {
		threshold=output->get_label(i);
		checked=check_classification(threshold);
		tp=checked[0];
		fp=checked[1];
		fn=checked[2];
		tn=checked[3];
		(*result)[i]=(DREAL) tp/(tp+fn)-fp/(fp+tn);
		delete[] checked;
	}
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_balance0()
{
	if (balance0!=0.) return balance0;

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT tn=checked[3];
	balance0=.5*((DREAL) tp/all_true+tn/all_false);

	delete[] checked;
	return balance0;
}

void CPerformanceMeasures::get_balance(DREAL** result, INT* num)
{
	if (!output)
		throw ShogunException("No output data given!\n");
	if (num_labels<1)
		throw ShogunException("Need at least one example!\n");

	*num=num_labels;
	size_t sz=sizeof(DREAL)*num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Could not allocate memory for WR accuracy!\n");

	INT* checked;
	INT tp;
	INT tn;
	DREAL threshold;

	for (INT i=0; i<num_labels; i++) {
		threshold=output->get_label(i);
		checked=check_classification(threshold);
		tp=checked[0];
		tn=checked[3];
		(*result)[i]=.5*((DREAL) tp/all_true+tn/all_false);
		delete[] checked;
	}
}

