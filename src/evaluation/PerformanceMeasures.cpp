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
: CSGObject(), m_true_labels(NULL), m_output(NULL), m_sortedROC(NULL)
{
	try {
		init(NULL, NULL);
	} catch (ShogunException(e)) {}
}

CPerformanceMeasures::CPerformanceMeasures(
	CLabels* true_labels, CLabels* output)
: CSGObject(), m_true_labels(NULL), m_output(NULL), m_sortedROC(NULL)
{
	init(true_labels, output);
}

CPerformanceMeasures::~CPerformanceMeasures()
{
	if (m_true_labels) SG_UNREF(m_true_labels);
	if (m_output) SG_UNREF(m_output);
	if (m_sortedROC) delete[] m_sortedROC;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::init(CLabels* true_labels, CLabels* output)
{
	m_all_true=0;
	m_all_false=0;
	m_num_labels=0;
	m_auROC=CMath::ALMOST_NEG_INFTY;
	m_accuracy0=CMath::ALMOST_NEG_INFTY;
	m_auPRC=CMath::ALMOST_NEG_INFTY;
	m_fmeasure0=CMath::ALMOST_NEG_INFTY;
	m_auDET=CMath::ALMOST_NEG_INFTY;
	m_cc0=CMath::ALMOST_NEG_INFTY;
	m_wracc0=CMath::ALMOST_NEG_INFTY;
	m_bal0=CMath::ALMOST_NEG_INFTY;

	if (!true_labels)
		throw ShogunException("No true labels given!\n");
	if (!output)
		throw ShogunException("No output given!\n");

	DREAL* labels=true_labels->get_labels(m_num_labels);
	if (m_num_labels<1) {
		delete[] labels;
		throw ShogunException("Need at least one example!\n");
	}
	if (m_num_labels!=output->get_num_labels()) {
		delete[] labels;
		throw ShogunException("Number of true labels and output labels differ!\n");
	}

	if (m_sortedROC) {
		delete[] m_sortedROC;
		m_sortedROC=NULL;
	}
	if (m_true_labels) {
		SG_UNREF(m_true_labels);
		m_true_labels=NULL;
	}
	if (m_output) {
		SG_UNREF(m_output);
		m_output=NULL;
	}

	for (INT i=0; i<m_num_labels; i++) {
		if (labels[i]==1.)
			m_all_true++;
		else if (labels[i]==-1.)
			m_all_false++;
		else {
			delete[] labels;
			throw ShogunException("Illegal true labels, not purely {-1, 1}!\n");
		}
	}
	delete[] labels;

	m_true_labels=true_labels;
	SG_REF(true_labels);
	m_output=output;
	SG_REF(output);

	create_sortedROC();
}

void CPerformanceMeasures::create_sortedROC()
{
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(INT)*m_num_labels;
	if (m_sortedROC) delete[] m_sortedROC;
	m_sortedROC=new INT[sz];
	if (!m_sortedROC)
		throw ShogunException("Couldn't allocate memory for sorted ROC index!\n");

	for (INT i=0; i<m_num_labels; i++) m_sortedROC[i]=i;
	DREAL* out=m_output->get_labels(m_num_labels);
	CMath::qsort_backward_index(out, m_sortedROC, m_num_labels);
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
	if (!m_true_labels)
		throw ShogunException("No true labels given!\n");
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	INT num_types=4;
	INT* types=new INT[num_types];
	for (INT i=0; i<num_types; i++) types[i]=0;

	for (INT i=0; i<m_num_labels; i++) {
		if (m_output->get_label(i)>=threshold) {
			if (m_true_labels->get_label(i)>0) types[0]++;
			else types[1]++;
		} else {
			if (m_true_labels->get_label(i)>0) types[2]++;
			else types[3]++;
		}
	}

	return types;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_ROC(DREAL** result, INT *dim, INT *num)
{
	*dim=m_num_labels+1;
	*num=2;
	compute_ROC(result);
}

void CPerformanceMeasures::compute_ROC(DREAL** result)
{
	if (!m_true_labels)
		throw ShogunException("No true labels given!\n");
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_all_true<1)
		throw ShogunException("Need at least one positive example in true labels!\n");
	if (m_all_false<1)
		throw ShogunException("Need at least one negative example in true labels!\n");

	// num_labels+1 due to point 1,1
	INT num_roc=m_num_labels+1;
	size_t sz=sizeof(DREAL)*num_roc*2;
	*result=(DREAL*) malloc(sz);
	if (!*result)
		throw ShogunException("Couldn't allocate memory for ROC result!\n");

	INT fp=0;
	INT tp=0;
	INT fp_prev=0;
	INT tp_prev=0;
	DREAL out_prev=CMath::ALMOST_NEG_INFTY;
	m_auROC=0.;
	INT i;

	for (i=0; i<m_num_labels; i++) {
		DREAL out=m_output->get_label(m_sortedROC[i]);
		if (out!=out_prev) {
			(*result)[i]=(DREAL) fp/m_all_false;
			(*result)[num_roc+i]=(DREAL) tp/m_all_true;
			m_auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);

			fp_prev=fp;
			tp_prev=tp;
			out_prev=out;
		}

		if (m_true_labels->get_label(m_sortedROC[i])==1) {
			tp++;
		} else {
			fp++;
		}
	}

	// calculate for 1,1
	(*result)[i]=(DREAL) fp/m_all_false;
	(*result)[num_roc+i]=(DREAL) tp/m_all_true;

	/* paper says:
	 * auROC+=trapezoid_area(1, fp_prev, 1, tp_prev)
	 * wrong? was meant for calculating with rates?
	 */
	m_auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);
	m_auROC/=m_all_true*m_all_false; // normalise
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_accuracy0()
{
	if (m_accuracy0!=CMath::ALMOST_NEG_INFTY) return m_accuracy0;

	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT tn=checked[3];
	m_accuracy0=(DREAL) (tp+tn)/m_num_labels;

	delete[] checked;
	return m_accuracy0;
}

void CPerformanceMeasures::compute_accuracyROC(DREAL** result, bool do_error)
{
	if (!m_true_labels)
		throw ShogunException("No true labels given!\n");
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*m_num_labels;
	*result=(DREAL*) malloc(sz);
	if (!*result)
		throw ShogunException("Couldn't allocate memory for accuracyROC!\n");

	DREAL out_prev=CMath::ALMOST_NEG_INFTY;
	INT tp=0;
	INT tn=m_all_false;

	for (INT i=0; i<m_num_labels; i++) {
		DREAL out=m_output->get_label(m_sortedROC[i]);
		if (out!=out_prev) {
			(*result)[i]=(DREAL) (tp+tn)/m_num_labels;
			if (do_error) (*result)[i]=1.-(*result)[i];
			out_prev=out;
		}

		if (m_true_labels->get_label(m_sortedROC[i])==1) {
			tp++;
		} else {
			tn--;
		}
	}
}

void CPerformanceMeasures::get_accuracyROC(DREAL** result, INT* num)
{
	*num=m_num_labels;
	compute_accuracyROC(result, false);
}

void CPerformanceMeasures::get_errorROC(DREAL** result, INT *num)
{
	*num=m_num_labels;
	compute_accuracyROC(result, true);
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_PRC(DREAL** result, INT *dim, INT *num)
{
	*dim=m_num_labels;
	*num=2;
	compute_PRC(result);
}

// FIXME: make as efficient as compute_ROC
void CPerformanceMeasures::compute_PRC(DREAL** result)
{
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*m_num_labels*2;
	*result=(DREAL*) malloc(sz);
	if (!*result)
		throw ShogunException("Couldn't allocate memory for PRC result!\n");

	INT* checked;
	INT tp;
	INT fp;
	DREAL threshold;

	for (INT i=0; i<m_num_labels; i++) {
		threshold=m_output->get_label(i);
		checked=check_classification(threshold);
		tp=checked[0];
		fp=checked[1];
		(*result)[i]=(DREAL) tp/m_all_true; // recall
		(*result)[m_num_labels+i]=(DREAL) tp/(tp+fp); // precision

		delete[] checked;
	}

	// sort by ascending recall
	CMath::qsort_index(*result, (*result)+m_num_labels, m_num_labels);

	// calculate auPRC
	m_auPRC=0.;
	for (INT i=0; i<m_num_labels-1; i++) {
		if ((*result)[1+i]==(*result)[i]) continue;
		m_auPRC+=trapezoid_area((*result)[1+i], (*result)[i], (*result)[1+m_num_labels+i], (*result)[m_num_labels+i]);
	}
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_fmeasurePRC(DREAL** result, INT* num)
{
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*m_num_labels;
	*result=(DREAL*) malloc(sz);
	if (!*result)
		throw ShogunException("Couldn't allocate memory for F-measure!\n");

	*num=m_num_labels;
	DREAL** prc=(DREAL**) malloc(sizeof(DREAL**));
	compute_PRC(prc);

	for (INT i=0; i<m_num_labels; i++) {
		(*result)[i]=2./(1./(*prc)[i+m_num_labels]+1./(*prc)[i]);
	}
	free(*prc);
	free(prc);
}

DREAL CPerformanceMeasures::get_fmeasure0()
{
	if (m_fmeasure0!=CMath::ALMOST_NEG_INFTY) return m_fmeasure0;

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT fp=checked[1];
	DREAL recall=(DREAL) tp/m_all_true;
	DREAL precision=(DREAL) tp/(tp+fp);
	m_fmeasure0=2./(1./precision+1./recall);

	delete[] checked;
	return m_fmeasure0;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_DET(DREAL** result, INT *dim, INT *num)
{
	*dim=m_num_labels;
	*num=2;
	compute_DET(result);
}

// FIXME: make as efficient as compute_ROC
void CPerformanceMeasures::compute_DET(DREAL** result)
{
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*m_num_labels*2;
	*result=(DREAL*) malloc(sz);
	if (!*result)
		throw ShogunException("Couldn't allocate memory for DET result!\n");

	INT* checked;
	INT fp;
	INT fn;
	DREAL threshold;

	for (INT i=0; i<m_num_labels; i++) {
		threshold=m_output->get_label(i);
		checked=check_classification(threshold);
		fp=checked[1];
		fn=checked[2];
		(*result)[i]=(DREAL) fp/m_all_false;
		(*result)[m_num_labels+i]=(DREAL) fn/m_all_false;

		delete[] checked;
	}

	// sort by ascending false positive rate
	CMath::qsort_index(*result, (*result)+m_num_labels, m_num_labels);

	// calculate auDET
	m_auDET=0.;
	for (INT i=0; i<m_num_labels-1; i++) {
		if ((*result)[1+i]==(*result)[i]) continue;
		m_auDET+=trapezoid_area((*result)[1+i], (*result)[i], (*result)[1+m_num_labels+i], (*result)[m_num_labels+i]);
	}
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_CC0()
{
	if (m_cc0!=CMath::ALMOST_NEG_INFTY) return m_cc0;

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT fp=checked[1];
	INT fn=checked[2];
	INT tn=checked[3];
	m_cc0=(tp*tn-fp*fn)/CMath::sqrt((DREAL) (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

	delete[] checked;
	return m_cc0;
}

void CPerformanceMeasures::get_CC(DREAL** result, INT* num)
{
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	*num=m_num_labels;
	size_t sz=sizeof(DREAL)*m_num_labels;
	*result=(DREAL*) malloc(sz);
	if (!result)
		throw ShogunException("Couldn't allocate memory for CC!\n");

	INT* checked;
	INT tp;
	INT fp;
	INT fn;
	INT tn;
	DREAL threshold;

	for (INT i=0; i<m_num_labels; i++) {
		threshold=m_output->get_label(i);
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

DREAL CPerformanceMeasures::get_WRAcc0()
{
	if (m_wracc0!=CMath::ALMOST_NEG_INFTY) return m_wracc0;

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT fp=checked[1];
	INT fn=checked[2];
	INT tn=checked[3];
	m_wracc0=(DREAL) tp/(tp+fn)-fp/(fp+tn);

	delete[] checked;
	return m_wracc0;
}

void CPerformanceMeasures::get_WRAcc(DREAL** result, INT* num)
{
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	*num=m_num_labels;
	size_t sz=sizeof(DREAL)*m_num_labels;
	*result=(DREAL*) malloc(sz);
	if (!*result)
		throw ShogunException("Couldn't allocate memory for WRAcc!\n");

	INT* checked;
	INT tp;
	INT fp;
	INT fn;
	INT tn;
	DREAL threshold;

	for (INT i=0; i<m_num_labels; i++) {
		threshold=m_output->get_label(i);
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

DREAL CPerformanceMeasures::get_BAL0()
{
	if (m_bal0!=CMath::ALMOST_NEG_INFTY) return m_bal0;

	INT* checked=check_classification(0);
	INT tp=checked[0];
	INT tn=checked[3];
	m_bal0=.5*((DREAL) tp/m_all_true+tn/m_all_false);

	delete[] checked;
	return m_bal0;
}

void CPerformanceMeasures::get_BAL(DREAL** result, INT* num)
{
	if (!m_output)
		throw ShogunException("No output data given!\n");
	if (m_num_labels<1)
		throw ShogunException("Need at least one example!\n");

	*num=m_num_labels;
	size_t sz=sizeof(DREAL)*m_num_labels;
	*result=(DREAL*) malloc(sz);
	if (!*result)
		throw ShogunException("Couldn't allocate memory for BAL!\n");

	INT* checked;
	INT tp;
	INT tn;
	DREAL threshold;

	for (INT i=0; i<m_num_labels; i++) {
		threshold=m_output->get_label(i);
		checked=check_classification(threshold);
		tp=checked[0];
		tn=checked[3];
		(*result)[i]=.5*((DREAL) tp/m_all_true+tn/m_all_false);
		delete[] checked;
	}
}

