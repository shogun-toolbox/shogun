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
	init_nolabels();
}

CPerformanceMeasures::CPerformanceMeasures(
	CLabels* true_labels, CLabels* output)
: CSGObject(), m_true_labels(NULL), m_output(NULL), m_sortedROC(NULL)
{
	init(true_labels, output);
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
	m_all_true=0;
	m_all_false=0;
	m_num_labels=0;
	m_auROC=CMath::ALMOST_NEG_INFTY;
	m_auPRC=CMath::ALMOST_NEG_INFTY;
	m_auDET=CMath::ALMOST_NEG_INFTY;
}

void CPerformanceMeasures::init(CLabels* true_labels, CLabels* output)
{
	init_nolabels();

	if (!true_labels)
		SG_ERROR("No true labels given!\n");
	if (!output)
		SG_ERROR("No output given!\n");

	DREAL* labels=true_labels->get_labels(m_num_labels);
	if (m_num_labels<1)
	{
		delete[] labels;
		SG_ERROR("Need at least one example!\n");
	}

	if (m_num_labels!=output->get_num_labels())
	{
		delete[] labels;
		SG_ERROR("Number of true labels and output labels differ!\n");
	}

	if (m_sortedROC)
	{
		delete[] m_sortedROC;
		m_sortedROC=NULL;
	}

	if (m_true_labels)
	{
		SG_UNREF(m_true_labels);
		m_true_labels=NULL;
	}

	if (m_output)
	{
		SG_UNREF(m_output);
		m_output=NULL;
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

	m_true_labels=true_labels;
	SG_REF(true_labels);
	m_output=output;
	SG_REF(output);
}

void CPerformanceMeasures::create_sortedROC()
{
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	size_t sz=sizeof(int32_t)*m_num_labels;
	if (m_sortedROC) delete[] m_sortedROC;
	m_sortedROC=new int32_t[sz];
	if (!m_sortedROC)
		SG_ERROR("Couldn't allocate memory for sorted ROC index!\n");

	for (int32_t i=0; i<m_num_labels; i++)
		m_sortedROC[i]=i;
	DREAL* out=m_output->get_labels(m_num_labels);
	CMath::qsort_backward_index(out, m_sortedROC, m_num_labels);
	delete[] out;
}

/////////////////////////////////////////////////////////////////////

template <class T> DREAL CPerformanceMeasures::trapezoid_area(T x1, T x2, T y1, T y2)
{
	DREAL base=CMath::abs(x1-x2);
	DREAL height_avg=0.5*(DREAL)(y1+y2);
	return base*height_avg;
}

void CPerformanceMeasures::compute_confusion_matrix(DREAL threshold, int32_t *tp, int32_t* fp, int32_t* fn, int32_t* tn)
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

void CPerformanceMeasures::get_ROC(DREAL** result, int32_t *num, int32_t *dim)
{
	*num=m_num_labels+1;
	*dim=2;
	compute_ROC(result);
}

void CPerformanceMeasures::compute_ROC(DREAL** result)
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
	size_t sz=sizeof(DREAL)*num_roc*2;

	DREAL* r=(DREAL*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for ROC result!\n");

	int32_t fp=0;
	int32_t tp=0;
	int32_t fp_prev=0;
	int32_t tp_prev=0;
	DREAL out_prev=CMath::ALMOST_NEG_INFTY;
	m_auROC=0.;
	int32_t i;

	for (i=0; i<m_num_labels; i++)
	{
		DREAL out=m_output->get_label(m_sortedROC[i]);
		if (out!=out_prev)
		{
			r[i]=(DREAL)fp/(DREAL)m_all_false;
			r[num_roc+i]=(DREAL)tp/(DREAL)m_all_true;
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
	r[i]=(DREAL)fp/(DREAL)(m_all_false);
	r[num_roc+i]=(DREAL)tp/DREAL(m_all_true);

	/* paper says:
	 * auROC+=trapezoid_area(1, fp_prev, 1, tp_prev)
	 * wrong? was meant for calculating with rates?
	 */
	m_auROC+=trapezoid_area(fp, fp_prev, tp, tp_prev);
	m_auROC/=(DREAL)(m_all_true*m_all_false); // normalise
	*result=r;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_PRC(DREAL** result, int32_t *num, int32_t *dim)
{
	*num=m_num_labels;
	*dim=2;
	compute_PRC(result);
}

// FIXME: make as efficient as compute_ROC
void CPerformanceMeasures::compute_PRC(DREAL** result)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*m_num_labels*2;
	DREAL* r=(DREAL*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for PRC result!\n");

	int32_t tp, fp;
	DREAL threshold;

	for (int32_t i=0; i<m_num_labels; i++)
	{
		threshold=m_output->get_label(i);
		compute_confusion_matrix(threshold, &tp, &fp, NULL, NULL);
		r[i]=(DREAL)tp/(DREAL)m_all_true; // recall
		r[m_num_labels+i]=(DREAL)tp/(DREAL)(tp+fp); // precision
	}

	// sort by ascending recall
	CMath::qsort_index(r, r+m_num_labels, m_num_labels);

	// calculate auPRC
	m_auPRC=0.;
	for (int32_t i=0; i<m_num_labels-1; i++)
	{
		if (r[1+i]==r[i])
			continue;
		m_auPRC+=trapezoid_area(r[1+i], r[i], r[1+m_num_labels+i], r[m_num_labels+i]);
	}

	*result=r;
}

/////////////////////////////////////////////////////////////////////

void CPerformanceMeasures::get_DET(DREAL** result, int32_t *num, int32_t *dim)
{
	*num=m_num_labels;
	*dim=2;
	compute_DET(result);
}

// FIXME: make as efficient as compute_ROC
void CPerformanceMeasures::compute_DET(DREAL** result)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	size_t sz=sizeof(DREAL)*m_num_labels*2;
	DREAL* r=(DREAL*) malloc(sz);
	if (!r)
		SG_ERROR("Couldn't allocate memory for DET result!\n");

	int32_t fp, fn;
	DREAL threshold;

	for (int32_t i=0; i<m_num_labels; i++)
	{
		threshold=m_output->get_label(i);
		compute_confusion_matrix(threshold, NULL, &fp, &fn, NULL);
		r[i]=(DREAL)fp/(DREAL)m_all_false;
		r[m_num_labels+i]=(DREAL)fn/(DREAL)m_all_false;
	}

	// sort by ascending false positive rate
	CMath::qsort_index(r, r+m_num_labels, m_num_labels);

	// calculate auDET
	m_auDET=0;
	for (int32_t i=0; i<m_num_labels-1; i++)
	{
		if (r[1+i]==r[i])
			continue;
		m_auDET+=trapezoid_area(r[1+i], r[i], r[1+m_num_labels+i], r[m_num_labels+i]);
	}

	*result=r;
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_accuracy(DREAL threshold)
{
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	int32_t tp, tn;

	compute_confusion_matrix(threshold, &tp, NULL, NULL, &tn);

	return (DREAL)(tp+tn)/(DREAL)m_num_labels;
}

void CPerformanceMeasures::compute_accuracy(
	DREAL** result, int32_t* num, int32_t* dim, bool do_error)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(DREAL)*m_num_labels*(*dim);
	DREAL* r=(DREAL*) malloc(sz);
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

void CPerformanceMeasures::get_all_accuracy(DREAL** result, int32_t* num, int32_t* dim)
{
	compute_accuracy(result, num, dim, false);
}

void CPerformanceMeasures::get_all_error(DREAL** result, int32_t *num, int32_t* dim)
{
	compute_accuracy(result, num, dim, true);
}

/////////////////////////////////////////////////////////////////////

DREAL CPerformanceMeasures::get_fmeasure(DREAL threshold)
{
	DREAL recall, precision;
	DREAL denominator;
	int32_t tp, fp;

	compute_confusion_matrix(threshold, &tp, &fp, NULL, NULL);

	if (m_all_true==0)
		return 0;
	else
		recall=(DREAL)tp/(DREAL)m_all_true;

	denominator=(DREAL)(tp+fp);
	if (denominator==0)
		return 0;
	else
		precision=(DREAL)tp/denominator;

	if (recall==0 && precision==0)
		return 0;
	else if (recall==0)
		return 2.0/(1/precision);
	else if (precision==0)
		return 2.0/(1/recall);
	else
		return 2.0/(1/precision+1/recall);
}

void CPerformanceMeasures::get_all_fmeasure(DREAL** result, int32_t* num, int32_t* dim)
{
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(DREAL)*m_num_labels*(*dim);
	DREAL* r=(DREAL*) malloc(sz);
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

DREAL CPerformanceMeasures::get_CC(DREAL threshold)
{
	int32_t tp, fp, fn, tn;
	DREAL radix;

	compute_confusion_matrix(threshold, &tp, &fp, &fn, &tn);

	radix=(DREAL)(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn);
	if (radix<=0)
		return 0;
	else
		return (DREAL)(tp*tn-fp*fn)/CMath::sqrt(radix);
}

void CPerformanceMeasures::get_all_CC(DREAL** result, int32_t* num, int32_t* dim)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(DREAL)*m_num_labels*(*dim);

	DREAL* r=(DREAL*) malloc(sz);
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

DREAL CPerformanceMeasures::get_WRAcc(DREAL threshold)
{
	int32_t tp, fp, fn, tn;
	DREAL denominator0, denominator1;

	compute_confusion_matrix(threshold, &tp, &fp, &fn, &tn);

	denominator0=(DREAL)(tp+fn);
	denominator1=(DREAL)(fp+tn);
	if (denominator0<=0 && denominator1<=0)
		return 0;
	else if (denominator0==0)
		return -(DREAL)fp/denominator1;
	else if (denominator1==0)
		return (DREAL)tp/denominator0;
	else
		return (DREAL)tp/denominator0-(DREAL)fp/denominator1;
}

void CPerformanceMeasures::get_all_WRAcc(DREAL** result, int32_t* num, int32_t* dim)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(DREAL)*m_num_labels*(*dim);

	DREAL* r=(DREAL*) malloc(sz);
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

DREAL CPerformanceMeasures::get_BAL(DREAL threshold)
{
	int32_t fp, fn;

	compute_confusion_matrix(threshold, NULL, &fp, &fn, NULL);

	if (m_all_true==0 && m_all_false==0) // actually a logical error
		return 0;
	else if (m_all_true==0)
		return 0.5*((DREAL)fp/(DREAL)m_all_false);
	else if (m_all_false==0)
		return 0.5*((DREAL)fn/(DREAL)m_all_true);
	else
		return 0.5*((DREAL)fp/(DREAL)m_all_false+(DREAL)fn/(DREAL)m_all_true);
}

void CPerformanceMeasures::get_all_BAL(DREAL** result, int32_t* num, int32_t* dim)
{
	if (!m_output)
		SG_ERROR("No output data given!\n");
	if (m_num_labels<1)
		SG_ERROR("Need at least one example!\n");

	*num=m_num_labels;
	*dim=2;
	size_t sz=sizeof(DREAL)*m_num_labels*(*dim);

	DREAL* r=(DREAL*) malloc(sz);
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
