/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include <string.h>

#include "lib/io.h"

#include "interface/SGInterface.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIPluginEstimate.h"

#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/Chi2Kernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/LinearByteKernel.h"
#include "kernel/LinearStringKernel.h"
#include "kernel/LinearWordKernel.h"
#include "kernel/WeightedDegreeStringKernel.h"
#include "kernel/WeightedDegreePositionStringKernel.h"
#include "kernel/FixedDegreeStringKernel.h"
#include "kernel/LocalityImprovedStringKernel.h"
#include "kernel/SimpleLocalityImprovedStringKernel.h"
#include "kernel/PolyKernel.h"
#include "kernel/CustomKernel.h"
#include "kernel/ConstKernel.h"
#include "kernel/PolyMatchWordKernel.h"
#include "kernel/PolyMatchStringKernel.h"
#include "kernel/LocalAlignmentStringKernel.h"
#include "kernel/WordMatchKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "kernel/CommUlongStringKernel.h"
#include "kernel/HistogramWordKernel.h"
#include "kernel/SalzbergWordKernel.h"
#include "kernel/GaussianKernel.h"
#include "kernel/GaussianShiftKernel.h"
#include "kernel/SigmoidKernel.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparsePolyKernel.h"
#include "kernel/SparseGaussianKernel.h"
#include "kernel/DiagKernel.h"
#include "kernel/MindyGramKernel.h"
#include "kernel/DistanceKernel.h"
#include "classifier/svm/SVM.h"


CGUIKernel::CGUIKernel(CSGInterface* ui_): CSGObject(), ui(ui_)
{
	kernel=NULL;
	initialized=false;
}

CGUIKernel::~CGUIKernel()
{
	delete kernel;
}

CKernel* CGUIKernel::get_kernel()
{
	return kernel;
}

#ifdef HAVE_MINDY
CKernel* CGUIKernel::create_mindygram(INT size, CHAR* meas_str, CHAR* norm_str, DREAL width, CHAR* param_str)
{
	CKernel* kern=new CMindyGramKernel(size, meast_str, width);
	if (!kern)
		SG_ERROR("Couldn't create MindyGramKernel with size %d, meas_str %s, width %f.\n", size, meas_str, width);
	else
		SG_DEBUG("created MindyGramKernel (%p) with size %d, meas_str %s, width %f.\n", kern, size, meas_str, width);

	ENormalizationType normalization=get_normalization_from_str(norm_str);
	kern->set_norm(normalization);
	kern->set_param(param_str);

	return kern;
}
#endif

CKernel* CGUIKernel::create_diag(INT size, DREAL diag)
{
	CKernel* kern=new CDiagKernel(size, diag);
	if (!kern)
		SG_ERROR("Couldn't create DiagKernel with size %d, diag %f.\n", size, diag);
	else
		SG_DEBUG("created DiagKernel (%p) with size %d, diag %f.\n", kern, size, diag);

	return kern;
}

CKernel* CGUIKernel::create_const(INT size, DREAL c)
{
	CKernel* kern=new CConstKernel(c);
	if (!kern)
		SG_ERROR("Couldn't create ConstKernel with c %f.\n", c);
	else
		SG_DEBUG("created ConstKernel (%p) with c %f.\n", kern, c);

	kern->set_cache_size(size);

	return kern;
}

CKernel* CGUIKernel::create_custom()
{
	CKernel* kern=new CCustomKernel();
	if (!kern)
		SG_ERROR("Couldn't create CustomKernel.\n");
	else
		SG_DEBUG("created CustomKernel (%p).\n", kern);

	return kern;
}


CKernel* CGUIKernel::create_gaussianshift(
	INT size, DREAL width, INT max_shift, INT shift_step)
{
	CKernel* kern=new CGaussianShiftKernel(size, width, max_shift, shift_step);
	if (!kern)
		SG_ERROR("Couldn't create GaussianShiftKernel with size %d, width %f, max_shift %d, shift_step %d.\n", size, width, max_shift, shift_step);
	else
		SG_DEBUG("created GaussianShiftKernel (%p) with size %d, width %f, max_shift %d, shift_step %d.\n", kern, size, width, max_shift, shift_step);

	return kern;
}

CKernel* CGUIKernel::create_sparsegaussian(INT size, DREAL width)
{
	CKernel* kern=new CSparseGaussianKernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create GaussianKernel with size %d, width %f.\n", size, width);
	else
		SG_DEBUG("created GaussianKernel (%p) with size %d, width %f.\n", kern, size, width);

	return kern;
}

CKernel* CGUIKernel::create_gaussian(INT size, DREAL width)
{
	CKernel* kern=new CGaussianKernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create GaussianKernel with size %d, width %f.\n", size, width);
	else
		SG_DEBUG("created GaussianKernel (%p) with size %d, width %f.\n", kern, size, width);

	return kern;
}

CKernel* CGUIKernel::create_sigmoid(INT size, DREAL gamma, DREAL coef0)
{
	CKernel* kern=new CSigmoidKernel(size, gamma, coef0);
	if (!kern)
		SG_ERROR("Couldn't create SigmoidKernel with size %d, gamma %f, coef0 %f.\n", size, gamma, coef0);
	else
		SG_DEBUG("created SigmoidKernel (%p) with size %d, gamma %f, coef0 %f.\n", kern, size, gamma, coef0);

	return kern;
}

CKernel* CGUIKernel::create_sparsepoly(
	INT size, INT degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CSparsePolyKernel(size, degree, inhomogene, normalize);
	if (!kern)
		SG_ERROR("Couldn't create SparsePolyKernel with size %d, degree %d, inhomoegene %d, normalize %d.\n", size, degree, inhomogene, normalize);
	else
		SG_DEBUG("created SparsePolyKernel with size %d, degree %d, inhomoegene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize);

	return kern;
}

CKernel* CGUIKernel::create_poly(
	INT size, INT degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyKernel(size, degree, inhomogene, normalize);
	if (!kern)
		SG_ERROR("Couldn't create PolyKernel with size %d, degree %d, inhomoegene %d, normalize %d.\n", size, degree, inhomogene, normalize);
	else
		SG_DEBUG("created PolyKernel (%p) with size %d, degree %d, inhomoegene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize);

	return kern;
}

CKernel* CGUIKernel::create_localityimprovedstring(
	INT size, INT length, INT inner_degree, INT outer_degree,
	EKernelType ktype)
{
	CKernel* kern=NULL;

	if (ktype==K_SIMPLELOCALITYIMPROVED)
	{
		kern=new CSimpleLocalityImprovedStringKernel(
			size, length, inner_degree, outer_degree);
	}
	else if (ktype==K_LOCALITYIMPROVED)
	{
		kern=new CLocalityImprovedStringKernel(
			size, length, inner_degree, outer_degree);
	}

	if (!kern)
		SG_ERROR("Couldn't create (Simple)LocalityImprovedStringKernel with size %d, length %d, inner_degree %d, outer_degree %d.\n", size, length, inner_degree, outer_degree);
	else
		SG_DEBUG("created (Simple)LocalityImprovedStringKernel with size %d, length %d, inner_degree %d, outer_degree %d.\n", kern, size, length, inner_degree, outer_degree);

	return kern;
}

CKernel* CGUIKernel::create_weighteddegreestring(
	INT size, INT order, INT max_mismatch, bool use_normalization,
	INT mkl_stepsize, bool block_computation, INT single_degree)
{
	DREAL* weights=get_weights(order, max_mismatch);

	INT i=0;
	if (single_degree>=0)
	{
		ASSERT(single_degree<order);
		for (i=0; i<order; i++)
		{
			if (i!=single_degree)
				weights[i]=0;
			else
				weights[i]=1;
		}
	}

	CKernel* kern=new CWeightedDegreeStringKernel(weights, order);
	if (!kern)
		SG_ERROR("Couldn't create WeightedDegreeStringKernel with size %d, order %d, max_mismatch %d, use_normalization %d, mkl_stepsize %d, block_computation %d, single_degree %f.\n", size, order, max_mismatch, use_normalization, mkl_stepsize, block_computation, single_degree);
	else
		SG_DEBUG("created WeightedDegreeStringKernel (%p) with size %d, order %d, max_mismatch %d, use_normalization %d, mkl_stepsize %d, block_computation %d, single_degree %f.\n", kern, size, order, max_mismatch, use_normalization, mkl_stepsize, block_computation, single_degree);

	((CWeightedDegreeStringKernel*) kern)->
		set_use_normalization(use_normalization);
	((CWeightedDegreeStringKernel*) kern)->
		set_use_block_computation(block_computation);
	((CWeightedDegreeStringKernel*) kern)->set_max_mismatch(max_mismatch);
	((CWeightedDegreeStringKernel*) kern)->set_mkl_stepsize(mkl_stepsize);
	((CWeightedDegreeStringKernel*) kern)->set_which_degree(single_degree);

	delete[] weights;
	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring(
	INT size, INT order, INT max_mismatch, INT length, INT center,
	DREAL step)
{
	INT i=0;
	INT* shifts=new INT[length];
	ASSERT(shifts);

	for (i=center; i<length; i++)
		shifts[i]=(int) floor(((DREAL) (i-center))/step);

	for (i=center-1; i>=0; i--)
		shifts[i]=(int) floor(((DREAL) (center-i))/step);

	for (i=0; i<length; i++)
	{
		if (shifts[i]>length)
			shifts[i]=length;
	}

	for (i=0; i<length; i++)
		SG_INFO( "shift[%i]=%i\n", i, shifts[i]);

	DREAL* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, weights, order, max_mismatch, shifts, length);
	if (!kern)
		SG_ERROR("Couldn't create WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d, center %d, step %f.\n", size, order, max_mismatch, length, center, step);
	else
		SG_DEBUG("created WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d, center %d, step %f.\n", kern, size, order, max_mismatch, length, center, step);

	delete[] weights;
	delete[] shifts;
	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring3(
	INT size, INT order, INT max_mismatch, INT* shifts, INT length,
	INT mkl_stepsize, DREAL* position_weights)
{
	DREAL* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, weights, order, max_mismatch, shifts, length, false, mkl_stepsize);
	if (!kern)
		SG_ERROR("Couldn't create WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d and position_weights (MKL stepsize: %d).\n", size, order, max_mismatch, length, mkl_stepsize);
	else
		SG_DEBUG("created WeightedDegreePositionStringKernel (%p) with size %d, order %d, max_mismatch %d, length %d and position_weights (MKL stepsize: %d).\n", kern, size, order, max_mismatch, length, mkl_stepsize);

	if (!position_weights)
	{
		position_weights=new DREAL[length];
		ASSERT(position_weights);
		for (INT i=0; i<length; i++)
			position_weights[i]=1.0/length;
	}
	((CWeightedDegreePositionStringKernel*) kern)->
		set_position_weights(position_weights, length);

	delete[] weights;
	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring2(
	INT size, INT order, INT max_mismatch, INT* shifts, INT length,
	bool use_normalization)
{
	DREAL* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, weights, order, max_mismatch, shifts, length, use_normalization);
	if (!kern)
		SG_ERROR("Couldn't create WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d, use_normalization %d.\n", size, order, max_mismatch, length, use_normalization);
	else
		SG_DEBUG("created WeightedDegreePositionStringKernel (%p) with size %d, order %d, max_mismatch %d, length %d, use_normalization %d.\n", kern, size, order, max_mismatch, length, use_normalization);

	delete[] weights;
	return kern;
}

DREAL* CGUIKernel::get_weights(INT order, INT max_mismatch)
{
	DREAL *weights=new DREAL[order*(1+max_mismatch)];
	ASSERT(weights);
	DREAL sum=0;
	INT i=0;

	for (i=0; i<order; i++)
	{
		weights[i]=order-i;
		sum+=weights[i];
	}
	for (i=0; i<order; i++)
		weights[i]/=sum;
	
	for (i=0; i<order; i++)
	{
		for (INT j=1; j<=max_mismatch; j++)
		{
			if (j<i+1)
			{
				INT nk=CMath::nchoosek(i+1, j);
				weights[i+j*order]=weights[i]/(nk*pow(3, j));
			}
			else
				weights[i+j*order]=0;
		}
	}

	return weights;
}


CKernel* CGUIKernel::create_localalignmentstring(INT size)
{
	CKernel* kern=new CLocalAlignmentStringKernel(size);
	if (!kern)
		SG_ERROR("Couldn't create LocalAlignmentStringKernel with size %d.\n", size);
	else
		SG_DEBUG("created LocalAlignmentStringKernel (%p) with size %d.\n", kern, size);

	return kern;
}

CKernel* CGUIKernel::create_fixeddegreestring(INT size, INT d)
{
	CKernel* kern=new CFixedDegreeStringKernel(size, d);
	if (!kern)
		SG_ERROR("Couldn't create FixedDegreeStringKernel with size %d and d %d.\n", size, d);
	else
		SG_DEBUG("created FixedDegreeStringKernel (%p) with size %d and d %d.\n", kern, size, d);

	return kern;
}

CKernel* CGUIKernel::create_chi2(INT size, DREAL width)
{
	CKernel* kern=new CChi2Kernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create Chi2Kernel with size %d and width %f.\n", size, width);
	else
		SG_DEBUG("created Chi2Kernel (%p) with size %d and width %f.\n", kern, size, width);

	return kern;
}

CKernel* CGUIKernel::create_commstring(
	INT size, bool use_sign, CHAR* norm_str, EKernelType ktype)
{
	ENormalizationType normalization=get_normalization_from_str(norm_str);

	CKernel* kern=NULL;
	if (ktype==K_COMMULONGSTRING)
		kern=new CCommUlongStringKernel(size, use_sign, normalization);
	else if (ktype==K_COMMWORDSTRING)
		kern=new CCommWordStringKernel(size, use_sign, normalization);
	else if (ktype==K_WEIGHTEDCOMMWORDSTRING)
		kern=new CWeightedCommWordStringKernel(size, use_sign, normalization);

	if (!kern)
		SG_ERROR("Couldn't create WeightedCommWord/CommWord/CommUlongStringKernel with size %d, use_sign  %d and normalization %d.\n", size, use_sign, normalization);
	else
		SG_DEBUG("created WeightedCommWord/CommWord/CommUlongStringKernel (%p) with size %d, use_sign  %d and normalization %d.\n", kern, size, use_sign, normalization);

	return kern;
}

ENormalizationType CGUIKernel::get_normalization_from_str(CHAR* str)
{
	if (!str)
		return FULL_NORMALIZATION;

	ENormalizationType norm=FULL_NORMALIZATION;

	if (strncmp(str, "NO", 2)==0)
	{
		norm=NO_NORMALIZATION;
		SG_INFO("Using no normalization.\n");
	}
	else if (strncmp(str, "SQRT", 4)==0)
	{
		norm=SQRT_NORMALIZATION;
		SG_INFO("Using sqrt normalization.\n");
	}
	else if (strncmp(str, "SQRTLEN", 7)==0)
	{
		norm=SQRTLEN_NORMALIZATION;
		SG_INFO("Using sqrt-len normalization.\n");
	}
	else if (strncmp(str, "LEN", 3)==0)
	{
		norm=LEN_NORMALIZATION;
		SG_INFO("Using len normalization.\n");
	}
	else if (strncmp(str, "SQLEN", 5)==0)
	{
		norm=SQLEN_NORMALIZATION;
		SG_INFO("Using squared len normalization.\n");
	}
	else if (strncmp(str, "FULL", 4)==0)
	{
		norm=FULL_NORMALIZATION;
		SG_INFO("Using full normalization.\n");
	}
	else
	{
		norm=FULL_NORMALIZATION;
		SG_INFO("Using default full normalization.\n");
	}

	return norm;
}

CKernel* CGUIKernel::create_wordmatch(INT size, INT d)
{
	CKernel* kern=new CWordMatchKernel(size, d);
	if (!kern)
		SG_ERROR("Couldn't create WordMatchKernel with size %d and d %d.\n", size, d);
	else
		SG_DEBUG("created WordMatchKernel (%p) with size %d and d %d.\n", kern, size, d);

	return kern;
}

CKernel* CGUIKernel::create_polymatchstring(
	INT size, INT degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyMatchStringKernel(size, degree, inhomogene, normalize);
	if (!kern)
		SG_ERROR("Couldn't create PolyMatchStringKernel with size %d, degree %d, inhomogene %d, normalize %d.\n", size, degree, inhomogene, normalize);
	else
		SG_DEBUG("created PolyMatchStringKernel (%p) with size %d, degree %d, inhomogene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize);

	return kern;
}

CKernel* CGUIKernel::create_polymatchword(
	INT size, INT degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyMatchWordKernel(size, degree, inhomogene, normalize);
	if (!kern)
		SG_ERROR("Couldn't create PolyMatchWordKernel with size %d, degree %d, inhomogene %d, normalize %d.\n", size, degree, inhomogene, normalize);
	else
		SG_DEBUG("created PolyMatchWordKernel (%p) with size %d, degree %d, inhomogene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize);

	return kern;
}

CKernel* CGUIKernel::create_salzbergword(INT size)
{

	SG_INFO("Getting estimator.\n");
	CPluginEstimate* estimator=ui->ui_pluginestimate->get_estimator();
	if (!estimator)
		SG_ERROR("No estimator set.\n");

	CKernel* kern=new CSalzbergWordKernel(size, estimator);
	if (!kern)
		SG_ERROR("Couldn't create HistogramWord with size %d.\n", size);
	else
		SG_DEBUG("created HistogramWord (%p) with size %d.\n", kern, size);

	// prior stuff
	SG_INFO("Getting labels.\n");
	CLabels* train_labels=ui->ui_labels->get_train_labels();
	if (!train_labels)
	{
		SG_INFO("Assign train labels first!\n");
		return NULL;
	}

	INT num_pos=0, num_neg=0;
	for (INT i=0; i<train_labels->get_num_labels(); i++)
	{
		if (train_labels->get_int_label(i)==1)
			num_pos++;
		if (train_labels->get_int_label(i)==-1)
			num_neg++;
	}
	SG_INFO("priors: pos=%1.3f (%i)  neg=%1.3f (%i)\n",
		(DREAL) num_pos/(num_pos+num_neg), num_pos,
		(DREAL) num_neg/(num_pos+num_neg), num_neg);

	((CSalzbergWordKernel*) kern)->set_prior_probs(
		(DREAL)num_pos/(num_pos+num_neg),
		(DREAL)num_neg/(num_pos+num_neg));

	return kern;
}

CKernel* CGUIKernel::create_histogramword(INT size)
{
	SG_INFO("Getting estimator.\n");
	CPluginEstimate* estimator=ui->ui_pluginestimate->get_estimator();
	if (!estimator)
		SG_ERROR("No estimator set.\n");

	CKernel* kern=new CHistogramWordKernel(size, estimator);
	if (!kern)
		SG_ERROR("Couldn't create HistogramWord with size %d.\n", size);
	else
		SG_DEBUG("created HistogramWord (%p) with size %d.\n", kern, size);

	return kern;
}

CKernel* CGUIKernel::create_linearbyte(INT size, DREAL scale)
{
	CKernel* kern=NULL;
	if (scale==-1)
		kern=new CLinearByteKernel(size, true);
	else
		kern=new CLinearByteKernel(size, false, scale);

	if (!kern)
		SG_ERROR("Couldn't create LinearByteKernel with size %d and scale %f.\n", size, scale);
	else
		SG_DEBUG("created LinearByteKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_linearword(INT size, DREAL scale)
{
	CKernel* kern=NULL;
	if (scale==-1)
		kern=new CLinearWordKernel(size, true);
	else
		kern=new CLinearWordKernel(size, false, scale);

	if (!kern)
		SG_ERROR("Couldn't create LinearWordKernel with size %d and scale %f.\n", size, scale);
	else
		SG_DEBUG("created LinearWordKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_linearstring(INT size, DREAL scale)
{
	CKernel* kern=NULL;
	if (scale==-1)
		kern=new CLinearStringKernel(size, true);
	else
		kern=new CLinearStringKernel(size, false, scale);

	if (!kern)
		SG_ERROR("Couldn't create LinearStringKernel with size %d and scale %f.\n", size, scale);
	else
		SG_DEBUG("created LinearStringKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_linear(INT size, DREAL scale)
{
	CKernel* kern=new CLinearKernel(size, scale);
	if (!kern)
		SG_ERROR("Couldn't create LinearKernel with size %d and scale %f.\n", size, scale);
	else
		SG_DEBUG("created LinearKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_sparselinear(INT size, DREAL scale)
{
	CKernel* kern=new CSparseLinearKernel(size, scale);
	if (!kern)
		SG_ERROR("Couldn't create SparseLinearKernel with size %d and scale %f.\n", size, scale);
	else
		SG_DEBUG("created SparseLinearKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_distance(INT size, DREAL width)
{
	CDistance* dist=ui->ui_distance->get_distance();
	if (!dist)
		SG_ERROR("No distance set for DistanceKernel.\n");

	CKernel* kern=new CDistanceKernel(size, width, dist);
	if (!kern)
		SG_ERROR("Couldn't create DistanceKernel with size %d and width %f.\n", size, width);
	else
		SG_DEBUG("created DistanceKernel (%p) with size %d and width %f.\n", kern, size, width);

	return kern;
}

CKernel* CGUIKernel::create_combined(
	INT size, bool append_subkernel_weights)
{
	CKernel* kern=new CCombinedKernel(size, append_subkernel_weights);
	if (!kern)
		SG_ERROR("Couldn't create CombinedKernel with size %d and append_subkernel_weights %d.\n", size, append_subkernel_weights);
	else
		SG_DEBUG("created CombinedKernel (%p) with size %d and append_subkernel_weights %d.\n", kern, size, append_subkernel_weights);

	return kern;
}

bool CGUIKernel::set_kernel(CKernel* kern)
{
	if (kern)
	{
		SG_DEBUG("deleting old kernel (%p).\n", kernel);
		delete kernel;
		kernel=kern;
		SG_DEBUG("set new kernel (%p).\n", kern);

		return true;
	}
	else
		return false;
}

bool CGUIKernel::load_kernel_init(CHAR* filename)
{
	bool result=false;
	if (kernel)
	{
		FILE* file=fopen(filename, "r");
		if (!file || !kernel->load_init(file))
			SG_ERROR("Reading from file %s failed!\n", filename);
		else
		{
			SG_INFO("Successfully read kernel init data from %s!\n", filename);
			initialized=true;
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		SG_ERROR("No kernel set!\n");

	return result;
}

bool CGUIKernel::save_kernel_init(CHAR* filename)
{
	bool result=false;

	if (kernel)
	{
		FILE* file=fopen(filename, "w");
		if (!file || !kernel->save_init(file))
			SG_ERROR("Writing to file %s failed!\n", filename);
		else
		{
			SG_INFO("Successfully written kernel init data into %s!\n", filename);
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		SG_ERROR("No kernel set!\n");

	return result;
}

bool CGUIKernel::init_kernel_optimization()
{
	kernel->set_precompute_matrix(false, false);

	CSVM* svm=(CSVM*) ui->ui_classifier->get_classifier();
	if (svm)
	{
		if (kernel->has_property(KP_LINADD))
		{
			INT num_sv=svm->get_num_support_vectors();
			INT* sv_idx=new INT[num_sv];
			DREAL* sv_weight=new DREAL[num_sv];
			
			for (INT i=0; i<num_sv; i++)
			{
				sv_idx[i]=svm->get_support_vector(i);
				sv_weight[i]=svm->get_alpha(i);
			}

			bool ret=kernel->init_optimization(num_sv, sv_idx, sv_weight);

			delete[] sv_idx;
			delete[] sv_weight;

			if (!ret)
				SG_ERROR("Initialization of kernel optimization failed\n");
			return ret;
		}
	}
	else
		SG_ERROR("Create SVM first!\n");

	return true;
}

bool CGUIKernel::delete_kernel_optimization()
{
	if (kernel && kernel->has_property(KP_LINADD) && kernel->get_is_initialized())
		kernel->delete_optimization();

	return true;
}


bool CGUIKernel::init_kernel(CHAR* target)
{
	if (!kernel)
	{
		SG_ERROR("No kernel available.\n");
		return false;
	}

	kernel->set_precompute_matrix(false, false);
	EFeatureClass k_fclass=kernel->get_feature_class();
	EFeatureType k_ftype=kernel->get_feature_type();

	if (!strncmp(target, "TRAIN", 5))
	{
		CFeatures* train=ui->ui_features->get_train_features();
		if (train)
		{
			EFeatureClass fclass=train->get_feature_class();
			EFeatureType ftype=train->get_feature_type();
			if ((k_fclass==fclass || k_fclass==C_ANY || fclass==C_ANY) &&
				(k_ftype==ftype || k_ftype==F_ANY || ftype==F_ANY))
			
			{
				kernel->init(train, train);
				initialized=true;
			}
			else
				SG_ERROR("Kernel can not process this train feature type: %d %d.\n", fclass, ftype);
		}
		else
			SG_ERROR("Assign train features first.\n");
	}
	else if (!strncmp(target, "TEST", 4))
	{
		CFeatures* train=ui->ui_features->get_train_features();
		CFeatures* test=ui->ui_features->get_test_features();
		if (test)
		{
			EFeatureClass fclass=test->get_feature_class();
			EFeatureType ftype=test->get_feature_type();
			if ((k_fclass==fclass || k_fclass==C_ANY || fclass==C_ANY) &&
				(k_ftype==ftype || k_ftype==F_ANY || ftype==F_ANY))
			
			{
				if (!initialized)
					SG_ERROR("Kernel not initialized with training examples.\n");
				else
				{
					SG_INFO("Initialising kernel with TEST DATA, train: %p test %p\n", train, test);
					// lhs -> always train_features; rhs -> always test_features
					kernel->init(train, test);
				}
			}
			else
				SG_ERROR("Kernel can not process this test feature type: %d %d.\n", fclass, ftype);
		}
		else
			SG_ERROR("Assign train and test features first.\n");
	}
	else
		SG_ERROR("Unknown target %s.\n", target);

	return true;
}

bool CGUIKernel::save_kernel(CHAR* filename)
{
	if (kernel && initialized)
	{
		if (!kernel->save(filename))
			SG_ERROR("Writing to file %s failed!\n", filename);
		else
		{
			SG_INFO("Successfully written kernel to \"%s\" !\n", filename);
			return true;
		}
	}
	else
		SG_ERROR("No kernel set / kernel not initialized!\n");

	return false;
}

bool CGUIKernel::add_kernel(CKernel* kern, DREAL weight)
{
	if (!kern)
		SG_ERROR("Given kernel to add is invalid.\n");

	if ((kernel==NULL) || (kernel && kernel->get_kernel_type()!=K_COMBINED))
	{
		delete kernel;
		kernel= new CCombinedKernel(20, false);
	}

	if (!kernel)
		SG_ERROR("Combined kernel object could not be created.\n");

	kern->set_combined_kernel_weight(weight);

	bool success=((CCombinedKernel*) kernel)->append_kernel(kern);
	if (success)
		((CCombinedKernel*) kernel)->list_kernels();
	else
		SG_ERROR("Adding of kernel failed.\n");

	return success;
}

bool CGUIKernel::clean_kernel()
{
	delete kernel;
	kernel=NULL;
	return true;
}

#ifdef USE_SVMLIGHT
bool CGUIKernel::resize_kernel_cache(INT size)
{
	if (!kernel)
		SG_ERROR("No kernel available.\n");

	kernel->resize_kernel_cache(size);
	return true;
}
#endif //USE_SVMLIGHT

bool CGUIKernel::set_optimization_type(CHAR* opt_type)
{
	EOptimizationType opt=SLOWBUTMEMEFFICIENT;
	if (!kernel)
		SG_ERROR("No kernel available.\n");

	if (strncmp(opt_type, "FASTBUTMEMHUNGRY", 16)==0)
	{
		SG_INFO("FAST METHOD selected\n");
		opt=FASTBUTMEMHUNGRY;
		kernel->set_optimization_type(opt);

		return true;
	}
	else if (strncmp(opt_type,"SLOWBUTMEMEFFICIENT", 19)==0)
	{
		SG_INFO("MEMORY EFFICIENT METHOD selected\n");
		opt=SLOWBUTMEMEFFICIENT;
		kernel->set_optimization_type(opt);

		return true;
	}
	else
		SG_ERROR("Wrong kernel optimization type.\n");

	return false;
}

bool CGUIKernel::del_kernel(CHAR* param)
{
	return false;
}
#endif
