/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Sebastian Henschel
 * Copyright (C) 2008-2009 Friedrich Miescher Laboratory of Max-Planck-Society
 */

#include "evaluation/PerformanceMeasures.h"

#include "lib/ShogunException.h"
#include "lib/Mathematics.h"

using namespace shogun;

CPerformanceMeasures::CPerformanceMeasures()
: CSGObject(), m_true_labels(NULL), m_output(NULL), m_sortedROC(NULL)
{
	init_nolabels();
	m_num_labels=0;
	m_all_true=0;
	m_all_false=0;
}

CPerformanceMeasures::CPerformanceMeasures(
	CLabels* true_labels, CLabels* output)
: CSGObject(), m_true_labels(NULL), m_output(NULL), m_sortedROC(NULL)
{
	m_num_labels=0;
	m_all_true=0;
	m_all_false=0;

	set_true_labels(true_labels);
	set_output(output);
}

CPerformanceMeasures::~CPerformanceMeasures()
{
	if (m_true_labels)
		SG_UNREF(m_true_labels);

	if (m_output)
		SG_UNREF(m_output);

	if (m_sortedROC)
		delete[] m_sortedROC;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::init_nolabels()
{
	delete[] m_sortedROC;
	m_sortedROC=NULL;

	m_auROC=CMath::ALMOST_NEG_INFTY;
	m_auPRC=CMath::ALMOST_NEG_INFTY;
	m_auDET=CMath::ALMOST_NEG_INFTY;
}

bool CPerformanceMeasures::set_true_labels(CLabels* true_labels)
{
	init_nolabels();

	if (!true_labels)
		SG_ERROR("No true labels given!\n");

	float64_t* labels=true_labels->get_labels(m_num_labels);
	if (m_num_labels<1)
	{
		delete[] labels;
		SG_ERROR("Need at least one example!\n");
	}

	if (m_output && m_num_labels!=m_output->get_num_labels())
	{
		delete[] labels;
		SG_ERROR("Number of true labels and output labels differ!\n");
	}

	for (int32_t i=0; i<m_num_labels; i++)
	{
		if (labels[i]==1)
			m_all_true++;
		else if (labels[i]==-1)
			m_all_false++;
		else
		{
			delete[] labels;
			SG_ERROR("Illegal true labels, not purely {-1, 1}!\n");
		}
	}
	delete[] labels;

	SG_UNREF(m_true_labels);
	m_true_labels=true_labels;
	SG_REF(m_true_labels);
	return true;
}

bool CPerformanceMeasures::set_output(CLabels* output)
{
	init_nolabels();

	if (!output)
		SG_ERROR("No output given!\n");

	if (m_true_labels && m_true_labels->get_num_labels() != output->get_num_labels())
		SG_ERROR("Number of true labels and output labels differ!\n");

	SG_UNREF(m_output);
	m_output=output;
	SG_REF(m_output);
	return true;
}

void CPerformanceMeasures::create_sortedROC()
{
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	delete[] m_sortedROC;
	m_sortedROC=new int32_t[m_num_labels];

	for (int32_t i=0; i<m_num_labels; i++)
		m_sortedROC[i]=i;
	float64_t* out=m_output->get_labels(m_num_labels);
	CMath::qsort_backward_index(out, m_sortedROC, m_num_labels);
	delete[] out;
}

/////////////////////////////////////////////////////////////////////

float64_t CPerformanceMeasures::trapezoid_area(
	float64_t x1, float64_t x2, float64_t y1, float64_t y2)
{
	float64_t base=CMath::abs(x1-x2);
	float64_t height_avg=0.5*(y1+y2);
	float64_t result=base*height_avg;

	if (result<0)
		SG_ERROR("Negative area - x1=%f x2=%f y1=%f y2=%f\n", x1,x2, y1,y2);
	return result;
}

void CPerformanceMeasures::compute_confusion_matrix(
	float64_t threshold, int32_t *tp, int32_t* fp, int32_t* fn, int32_t* tn)
{
	if (!m_true_labels)
		SG_ERROR("No true labels given!\n");
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	if (tp)
		*tp=0;
	if (fp)
		*fp=0;
	if (fn)
		*fn=0;
	if (tn)
		*tn=0;

	for (int32_t i=0; i<m_num_labels; i++)
	{
		if (m_output->get_label(i)>=threshold)
		{
			if (m_true_labels->get_label(i)>0)
			{
				if (tp)
					(*tp)++;
			}
			else
			{
				if (fp)
					(*fp)++;
			}
		}
		else
		{
			if (m_true_labels->get_label(i)>0)
			{
				if (fn)
					(*fn)++;
			}
			else
			{
				if (tn)
					(*tn)++;
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_ROC(
	float64_t** result, int32_t *num, int32_t *dim)
{
	*num=m_num_labels+1;
	*dim=2;
	compute_ROC(result);
}

void CPerformanceMeasures::compute_ROC(float64_t** result)
{
	if (!m_true_labels)
		SG_ERROR("No true labels given!\n");
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_all_true<1)
		SG_ERROR("Need at least one positive example in true labels!\n");
	if (m_all_false<1)
		SG_ERROR("Need at least one negative example in true labels!\n");

	if (!m_sortedROC)
		create_sortedROC();

	// num_labels+1 due to point 1,1
	int32_t num_roc=m_num_labels+1;
	size_t sz=sizeof(float64_t)*num_roc*2;

	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for ROC result!\n");

	int32_t fp=0;
	int32_t tp=0;
	int32_t fp_prev=0;
	int32_t tp_prev=0;
	float64_t out_prev=CMath::ALMOST_NEG_INFTY;
	m_auROC=0.;
	int32_t i;

	for (i=0; i<m_num_labels; i++)
	{
		float64_t out=m_output->get_label(m_sortedROC[i]);
		if (out!=out_prev)
		{
			r[i]=float64_t(fp)/m_all_false;
			r[num_roc+i]=float64_t(tp)/m_all_true;
			m_auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);

			fp_prev=fp;
			tp_prev=tp;
			out_prev=out;
		}

		if (m_true_labels->get_label(m_sortedROC[i])==1)
			tp++;
		else
			fp++;
	}

	// calculate for 1,1
	r[i]=float64_t(fp)/m_all_false;
	r[num_roc+i]=float64_t(tp)/m_all_true;

	/* paper says:
	 * auROC+=trapezoid_area(1, fp_prev, 1, tp_prev)
	 * wrong? was meant for calculating with rates?
	 */
	m_auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);
	m_auROC/=float64_t(m_all_true)*m_all_false; // normalise
	*result=r;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_PRC(
	float64_t** result, int32_t *num, int32_t *dim)
{
	*num=m_num_labels;
	*dim=2;
	compute_PRC(result);
}

// FIXME: make as efficient as compute_ROC
void CPerformanceMeasures::compute_PRC(float64_t** result)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	size_t sz=sizeof(float64_t)*m_num_labels*2;
	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for PRC result!\n");

	int32_t tp, fp;
	float64_t threshold;

	for (int32_t i=0; i<m_num_labels; i++)
	{
		threshold=m_output->get_label(i);
		compute_confusion_matrix(threshold, &tp, &fp, NULL, NULL);
		r[i]=float64_t(tp)/m_all_true; // recall
		r[m_num_labels+i]=float64_t(tp)/(float64_t(tp)+fp); // precision
	}

	// sort by ascending recall
	CMath::qsort_index(r, r+m_num_labels, m_num_labels);

	// calculate auPRC
	m_auPRC=0.;
	for (int32_t i=0; i<m_num_labels-1; i++)
	{
		if (r[1+i]==r[i])
			continue;
		m_auPRC+=trapezoid_area(
			r[1+i], r[i], r[1+m_num_labels+i], r[m_num_labels+i]);
	}

	*result=r;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_DET(
	float64_t** result, int32_t *num, int32_t *dim)
{
	*num=m_num_labels;
	*dim=2;
	compute_DET(result);
}

// FIXME: make as efficient as compute_ROC
void CPerformanceMeasures::compute_DET(float64_t** result)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	size_t sz=sizeof(float64_t)*m_num_labels*2;
	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for DET result!\n");

	int32_t fp, fn;
	float64_t threshold;

	for (int32_t i=0; i<m_num_labels; i++)
	{
		threshold=m_output->get_label(i);
		compute_confusion_matrix(threshold, NULL, &fp, &fn, NULL);
		r[i]=float64_t(fp)/m_all_false;
		r[m_num_labels+i]=float64_t(fn)/m_all_false;
	}

	// sort by ascending false positive rate
	CMath::qsort_index(r, r+m_num_labels, m_num_labels);

	// calculate auDET
	m_auDET=0;
	for (int32_t i=0; i<m_num_labels-1; i++)
	{
		if (r[1+i]==r[i])
			continue;
		m_auDET+=trapezoid_area(
			r[1+i], r[i], r[1+m_num_labels+i], r[m_num_labels+i]);
	}

	*result=r;
}

/////////////////////////////////////////////////////////////////////

float64_t CPerformanceMeasures::get_accuracy(float64_t threshold)
{
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	int32_t tp, tn;

	compute_confusion_matrix(threshold, &tp, NULL, NULL, &tn);

	return (float64_t(tp)+tn)/m_num_labels;
}

void CPerformanceMeasures::compute_accuracy(
	float64_t** result, int32_t* num, int32_t* dim, bool do_error)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(float64_t)*m_num_labels*(*dim);
	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for all accuracy points!\n");

	for (int32_t i=0; i<m_num_labels; i++)
	{
		r[i]=m_output->get_label(i);
		if (do_error)
			r[i+m_num_labels]=1.0-get_accuracy(r[i]);
		else
			r[i+m_num_labels]=get_accuracy(r[i]);
	}

	CMath::qsort_index(r, r+m_num_labels, m_num_labels);
	*result=r;
}

void CPerformanceMeasures::get_all_accuracy(
	float64_t** result, int32_t* num, int32_t* dim)
{
	compute_accuracy(result, num, dim, false);
}

void CPerformanceMeasures::get_all_error(
	float64_t** result, int32_t *num, int32_t* dim)
{
	compute_accuracy(result, num, dim, true);
}

/////////////////////////////////////////////////////////////////////

float64_t CPerformanceMeasures::get_fmeasure(float64_t threshold)
{
	float64_t recall, precision;
	float64_t denominator;
	int32_t tp, fp;

	compute_confusion_matrix(threshold, &tp, &fp, NULL, NULL);

	if (m_all_true==0)
		return 0;
	else
		recall=float64_t(tp)/m_all_true;

	denominator=float64_t(tp)+fp;
	if (denominator==0)
		return 0;
	else
		precision=float64_t(tp)/denominator;

	if (recall==0 && precision==0)
		return 0;
	else if (recall==0)
		return 2.0/(1.0/precision);
	else if (precision==0)
		return 2.0/(1.0/recall);
	else
		return 2.0/(1.0/precision+1.0/recall);
}

void CPerformanceMeasures::get_all_fmeasure(
	float64_t** result, int32_t* num, int32_t* dim)
{
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(float64_t)*m_num_labels*(*dim);
	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for all F-measure points!\n");

	for (int32_t i=0; i<m_num_labels; i++) {
		r[i]=m_output->get_label(i);
		r[i+m_num_labels]=get_fmeasure(r[i]);
	}

	CMath::qsort_index(r, r+m_num_labels, m_num_labels);
	*result=r;
}

/////////////////////////////////////////////////////////////////////

float64_t CPerformanceMeasures::get_CC(float64_t threshold)
{
	int32_t tp, fp, fn, tn;
	float64_t radix;

	compute_confusion_matrix(threshold, &tp, &fp, &fn, &tn);

	radix=(float64_t(tp)+fp)*(float64_t(tp)+fn)*(float64_t(tn)+fp)*(float64_t(tn)+fn);
	if (radix<=0)
		return 0;
	else
		return (float64_t(tp)*tn-float64_t(fp)*fn)/CMath::sqrt(radix);
}

void CPerformanceMeasures::get_all_CC(
	float64_t** result, int32_t* num, int32_t* dim)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(float64_t)*m_num_labels*(*dim);

	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for all CC points!\n");

	for (int32_t i=0; i<m_num_labels; i++)
	{
		r[i]=m_output->get_label(i);
		r[i+m_num_labels]=get_CC(r[i]);
	}

	CMath::qsort_index(r, r+m_num_labels, m_num_labels);
	*result=r;
}

/////////////////////////////////////////////////////////////////////

float64_t CPerformanceMeasures::get_WRAcc(float64_t threshold)
{
	int32_t tp, fp, fn, tn;
	float64_t denominator0, denominator1;

	compute_confusion_matrix(threshold, &tp, &fp, &fn, &tn);

	denominator0=float64_t(tp)+fn;
	denominator1=float64_t(fp)+tn;
	if (denominator0<=0 && denominator1<=0)
		return 0;
	else if (denominator0==0)
		return -fp/denominator1;
	else if (denominator1==0)
		return tp/denominator0;
	else
		return tp/denominator0-fp/denominator1;
}

void CPerformanceMeasures::get_all_WRAcc(
	float64_t** result, int32_t* num, int32_t* dim)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(float64_t)*m_num_labels*(*dim);

	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for all WRAcc points!\n");

	for (int32_t i=0; i<m_num_labels; i++)
	{
		r[i]=m_output->get_label(i);
		r[i+m_num_labels]=get_WRAcc(r[i]);
	}

	CMath::qsort_index(r, r+m_num_labels, m_num_labels);
	*result=r;
}

/////////////////////////////////////////////////////////////////////

float64_t CPerformanceMeasures::get_BAL(float64_t threshold)
{
	int32_t fp, fn;

	compute_confusion_matrix(threshold, NULL, &fp, &fn, NULL);

	if (m_all_true==0 && m_all_false==0) // actually a logical error
		return 0;
	else if (m_all_true==0)
		return 0.5*fp/m_all_false;
	else if (m_all_false==0)
		return 0.5*fn/m_all_true;
	else
		return 0.5*(float64_t(fp)/m_all_false+float64_t(fn)/m_all_true);
}

void CPerformanceMeasures::get_all_BAL(float64_t** result, int32_t* num, int32_t* dim)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(float64_t)*m_num_labels*(*dim);

	float64_t* r=(float64_t*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for all BAL points!\n");

	for (int32_t i=0; i<m_num_labels; i++)
	{
		r[i]=m_output->get_label(i);
		r[i+m_num_labels]=get_BAL(r[i]);
	}

	CMath::qsort_index(r, r+m_num_labels, m_num_labels);
	*result=r;
}
