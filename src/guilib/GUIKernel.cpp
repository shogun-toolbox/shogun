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
#include "kernel/PolyMatchWordStringKernel.h"
#include "kernel/PolyMatchStringKernel.h"
#include "kernel/LocalAlignmentStringKernel.h"
#include "kernel/MatchWordStringKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "kernel/CommUlongStringKernel.h"
#include "kernel/HistogramWordStringKernel.h"
#include "kernel/SalzbergWordStringKernel.h"
#include "kernel/GaussianKernel.h"
#include "kernel/GaussianShiftKernel.h"
#include "kernel/SigmoidKernel.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparsePolyKernel.h"
#include "kernel/SparseGaussianKernel.h"
#include "kernel/DiagKernel.h"
#include "kernel/MindyGramKernel.h"
#include "kernel/OligoKernel.h"
#include "kernel/DistanceKernel.h"

#include "kernel/AvgDiagKernelNormalizer.h"
#include "kernel/FirstElementKernelNormalizer.h"
#include "kernel/IdentityKernelNormalizer.h"
#include "kernel/SqrtDiagKernelNormalizer.h"

#include "classifier/svm/SVM.h"


CGUIKernel::CGUIKernel(CSGInterface* ui_)
: CSGObject(), ui(ui_)
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
CKernel* CGUIKernel::create_mindygram(
	int32_t size, char* meas_str, char* norm_str, float64_t width,
	char* param_str)
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

CKernel* CGUIKernel::create_oligo(int32_t size, int32_t k, float64_t width)
{
	CKernel* kern=new COligoKernel(size, k, width);
	SG_DEBUG("created OligoKernel (%p) with size %d, k %d, width %f.\n", kern, size, k, width);

	return kern;
}

CKernel* CGUIKernel::create_diag(int32_t size, float64_t diag)
{
	CKernel* kern=new CDiagKernel(size, diag);
	if (!kern)
		SG_ERROR("Couldn't create DiagKernel with size %d, diag %f.\n", size, diag);
	else
		SG_DEBUG("created DiagKernel (%p) with size %d, diag %f.\n", kern, size, diag);

	return kern;
}

CKernel* CGUIKernel::create_const(int32_t size, float64_t c)
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
	int32_t size, float64_t width, int32_t max_shift, int32_t shift_step)
{
	CKernel* kern=new CGaussianShiftKernel(size, width, max_shift, shift_step);
	if (!kern)
		SG_ERROR("Couldn't create GaussianShiftKernel with size %d, width %f, max_shift %d, shift_step %d.\n", size, width, max_shift, shift_step);
	else
		SG_DEBUG("created GaussianShiftKernel (%p) with size %d, width %f, max_shift %d, shift_step %d.\n", kern, size, width, max_shift, shift_step);

	return kern;
}

CKernel* CGUIKernel::create_sparsegaussian(int32_t size, float64_t width)
{
	CKernel* kern=new CSparseGaussianKernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create GaussianKernel with size %d, width %f.\n", size, width);
	else
		SG_DEBUG("created GaussianKernel (%p) with size %d, width %f.\n", kern, size, width);

	return kern;
}

CKernel* CGUIKernel::create_gaussian(int32_t size, float64_t width)
{
	CKernel* kern=new CGaussianKernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create GaussianKernel with size %d, width %f.\n", size, width);
	else
		SG_DEBUG("created GaussianKernel (%p) with size %d, width %f.\n", kern, size, width);

	return kern;
}

CKernel* CGUIKernel::create_sigmoid(
	int32_t size, float64_t gamma, float64_t coef0)
{
	CKernel* kern=new CSigmoidKernel(size, gamma, coef0);
	if (!kern)
		SG_ERROR("Couldn't create SigmoidKernel with size %d, gamma %f, coef0 %f.\n", size, gamma, coef0);
	else
		SG_DEBUG("created SigmoidKernel (%p) with size %d, gamma %f, coef0 %f.\n", kern, size, gamma, coef0);

	return kern;
}

CKernel* CGUIKernel::create_sparsepoly(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CSparsePolyKernel(size, degree, inhomogene);
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());
	SG_DEBUG("created SparsePolyKernel with size %d, degree %d, inhomogene %d normalize %d.\n", kern, size, degree, inhomogene, normalize);

	return kern;
}

CKernel* CGUIKernel::create_poly(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyKernel(size, degree, inhomogene);
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());
	SG_DEBUG("created PolyKernel (%p) with size %d, degree %d, inhomogene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize);

	return kern;
}

CKernel* CGUIKernel::create_localityimprovedstring(
	int32_t size, int32_t length, int32_t inner_degree, int32_t outer_degree,
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
	int32_t size, int32_t order, int32_t max_mismatch, bool use_normalization,
	int32_t mkl_stepsize, bool block_computation, int32_t single_degree)
{
	float64_t* weights=get_weights(order, max_mismatch);

	int32_t i=0;
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

	SG_DEBUG("created WeightedDegreeStringKernel (%p) with size %d, order %d, max_mismatch %d, use_normalization %d, mkl_stepsize %d, block_computation %d, single_degree %f.\n", kern, size, order, max_mismatch, mkl_stepsize, block_computation, single_degree);

	if (!use_normalization)
		kern->set_normalizer(new CIdentityKernelNormalizer());
		
	((CWeightedDegreeStringKernel*) kern)->
		set_use_block_computation(block_computation);
	((CWeightedDegreeStringKernel*) kern)->set_max_mismatch(max_mismatch);
	((CWeightedDegreeStringKernel*) kern)->set_mkl_stepsize(mkl_stepsize);
	((CWeightedDegreeStringKernel*) kern)->set_which_degree(single_degree);

	delete[] weights;
	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring(
	int32_t size, int32_t order, int32_t max_mismatch, int32_t length,
	int32_t center, float64_t step)
{
	int32_t i=0;
	int32_t* shifts=new int32_t[length];

	for (i=center; i<length; i++)
		shifts[i]=(int32_t) floor(((float64_t) (i-center))/step);

	for (i=center-1; i>=0; i--)
		shifts[i]=(int32_t) floor(((float64_t) (center-i))/step);

	for (i=0; i<length; i++)
	{
		if (shifts[i]>length)
			shifts[i]=length;
	}

	for (i=0; i<length; i++)
		SG_INFO( "shift[%i]=%i\n", i, shifts[i]);

	float64_t* weights=get_weights(order, max_mismatch);

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
	int32_t size, int32_t order, int32_t max_mismatch, int32_t* shifts,
	int32_t length, int32_t mkl_stepsize, float64_t* position_weights)
{
	float64_t* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, weights, order, max_mismatch, shifts, length, mkl_stepsize);
	kern->set_normalizer(new CIdentityKernelNormalizer());

	SG_DEBUG("created WeightedDegreePositionStringKernel (%p) with size %d, order %d, max_mismatch %d, length %d and position_weights (MKL stepsize: %d).\n", kern, size, order, max_mismatch, length, mkl_stepsize);

	if (!position_weights)
	{
		position_weights=new float64_t[length];
		for (int32_t i=0; i<length; i++)
			position_weights[i]=1.0/length;
	}
	((CWeightedDegreePositionStringKernel*) kern)->
		set_position_weights(position_weights, length);

	delete[] weights;
	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring2(
	int32_t size, int32_t order, int32_t max_mismatch, int32_t* shifts,
	int32_t length, bool use_normalization)
{
	float64_t* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, weights, order, max_mismatch, shifts, length);
	if (!use_normalization)
		kern->set_normalizer(new CIdentityKernelNormalizer());


	SG_DEBUG("created WeightedDegreePositionStringKernel (%p) with size %d, order %d, max_mismatch %d, length %d, use_normalization %d.\n", kern, size, order, max_mismatch, length, use_normalization);

	delete[] weights;
	return kern;
}

float64_t* CGUIKernel::get_weights(int32_t order, int32_t max_mismatch)
{
	float64_t *weights=new float64_t[order*(1+max_mismatch)];
	float64_t sum=0;
	int32_t i=0;

	for (i=0; i<order; i++)
	{
		weights[i]=order-i;
		sum+=weights[i];
	}
	for (i=0; i<order; i++)
		weights[i]/=sum;
	
	for (i=0; i<order; i++)
	{
		for (int32_t j=1; j<=max_mismatch; j++)
		{
			if (j<i+1)
			{
				int32_t nk=CMath::nchoosek(i+1, j);
				weights[i+j*order]=weights[i]/(nk*CMath::pow(3, j));
			}
			else
				weights[i+j*order]=0;
		}
	}

	return weights;
}


CKernel* CGUIKernel::create_localalignmentstring(int32_t size)
{
	CKernel* kern=new CLocalAlignmentStringKernel(size);
	if (!kern)
		SG_ERROR("Couldn't create LocalAlignmentStringKernel with size %d.\n", size);
	else
		SG_DEBUG("created LocalAlignmentStringKernel (%p) with size %d.\n", kern, size);

	return kern;
}

CKernel* CGUIKernel::create_fixeddegreestring(int32_t size, int32_t d)
{
	CKernel* kern=new CFixedDegreeStringKernel(size, d);
	if (!kern)
		SG_ERROR("Couldn't create FixedDegreeStringKernel with size %d and d %d.\n", size, d);
	else
		SG_DEBUG("created FixedDegreeStringKernel (%p) with size %d and d %d.\n", kern, size, d);

	return kern;
}

CKernel* CGUIKernel::create_chi2(int32_t size, float64_t width)
{
	CKernel* kern=new CChi2Kernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create Chi2Kernel with size %d and width %f.\n", size, width);
	else
		SG_DEBUG("created Chi2Kernel (%p) with size %d and width %f.\n", kern, size, width);

	return kern;
}

CKernel* CGUIKernel::create_commstring(
	int32_t size, bool use_sign, char* norm_str, EKernelType ktype)
{
	CKernel* kern=NULL;
	if (ktype==K_COMMULONGSTRING)
		kern=new CCommUlongStringKernel(size, use_sign);
	else if (ktype==K_COMMWORDSTRING)
		kern=new CCommWordStringKernel(size, use_sign);
	else if (ktype==K_WEIGHTEDCOMMWORDSTRING)
		kern=new CWeightedCommWordStringKernel(size, use_sign);

	SG_DEBUG("created WeightedCommWord/CommWord/CommUlongStringKernel (%p) with size %d, use_sign  %d.\n", kern, size, use_sign);

	return kern;
}

CKernel* CGUIKernel::create_matchwordstring(
	int32_t size, int32_t d, bool normalize)
{
	CKernel* kern=new CMatchWordStringKernel(size, d);
	SG_DEBUG("created MatchWordStringKernel (%p) with size %d and d %d.\n", kern, size, d);
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());

	return kern;
}

CKernel* CGUIKernel::create_polymatchstring(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyMatchStringKernel(size, degree, inhomogene);
	SG_DEBUG("created PolyMatchStringKernel (%p) with size %d, degree %d, inhomogene %d normalize %d.\n", kern, size, degree, inhomogene, normalize);
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());

	return kern;
}

CKernel* CGUIKernel::create_polymatchwordstring(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyMatchWordStringKernel(size, degree, inhomogene);
	SG_DEBUG("created PolyMatchWordStringKernel (%p) with size %d, degree %d, inhomogene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize);
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());

	return kern;
}

CKernel* CGUIKernel::create_salzbergword(int32_t size)
{
	SG_INFO("Getting estimator.\n");
	CPluginEstimate* estimator=ui->ui_pluginestimate->get_estimator();
	if (!estimator)
		SG_ERROR("No estimator set.\n");

	CKernel* kern=new CSalzbergWordStringKernel(size, estimator);
	if (!kern)
		SG_ERROR("Couldn't create SalzbergWordString with size %d.\n", size);
	else
		SG_DEBUG("created SalzbergWordString (%p) with size %d.\n", kern, size);

/*
	// prior stuff
	SG_INFO("Getting labels.\n");
	CLabels* train_labels=ui->ui_labels->get_train_labels();
	if (!train_labels)
	{
		SG_INFO("Assign train labels first!\n");
		return NULL;
	}
	((CSalzbergWordStringKernel *) kern)->set_prior_probs_from_labels(train_labels);
*/

	return kern;
}

CKernel* CGUIKernel::create_histogramword(int32_t size)
{
	SG_INFO("Getting estimator.\n");
	CPluginEstimate* estimator=ui->ui_pluginestimate->get_estimator();
	if (!estimator)
		SG_ERROR("No estimator set.\n");

	CKernel* kern=new CHistogramWordStringKernel(size, estimator);
	if (!kern)
		SG_ERROR("Couldn't create HistogramWordString with size %d.\n", size);
	else
		SG_DEBUG("created HistogramWordString (%p) with size %d.\n", kern, size);

	return kern;
}

CKernel* CGUIKernel::create_linearbyte(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CLinearByteKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));
	SG_DEBUG("created LinearByteKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_linearword(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CLinearWordKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));
	SG_DEBUG("created LinearWordKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_linearstring(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=NULL;
	kern=new CLinearStringKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));

	SG_DEBUG("created LinearStringKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_linear(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CLinearKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));

	SG_DEBUG("created LinearKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_sparselinear(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CSparseLinearKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));

	SG_DEBUG("created SparseLinearKernel (%p) with size %d and scale %f.\n", kern, size, scale);

	return kern;
}

CKernel* CGUIKernel::create_distance(int32_t size, float64_t width)
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
	int32_t size, bool append_subkernel_weights)
{
	CKernel* kern=new CCombinedKernel(size, append_subkernel_weights);
	if (!kern)
		SG_ERROR("Couldn't create CombinedKernel with size %d and append_subkernel_weights %d.\n", size, append_subkernel_weights);
	else
		SG_DEBUG("created CombinedKernel (%p) with size %d and append_subkernel_weights %d.\n", kern, size, append_subkernel_weights);

	return kern;
}

bool CGUIKernel::set_normalization(char* normalization, float64_t c)
{
	CKernel* k=kernel;

	if (k && k->get_kernel_type()==K_COMBINED)
		k=((CCombinedKernel*) kernel)->get_last_kernel();

	if (!k)
		SG_ERROR("No kernel available.\n");

	if (strncmp(normalization, "IDENTITY", 8)==0)
	{
		SG_INFO("Identity Normalization (==NO NORMALIZATION) selected\n");
		return k->set_normalizer(new CIdentityKernelNormalizer());
	}
	else if (strncmp(normalization,"AVGDIAG", 7)==0)
	{
		SG_INFO("Average Kernel Diagonal Normalization selected\n");
		return k->set_normalizer(new CAvgDiagKernelNormalizer(c));
	}
	else if (strncmp(normalization,"SQRTDIAG", 8)==0)
	{
		SG_INFO("Sqrt Diagonal Normalization selected\n");
		return k->set_normalizer(new CSqrtDiagKernelNormalizer());
	}
	else if (strncmp(normalization,"FIRSTELEMENT", 12)==0)
	{
		SG_INFO("First Element Normalization selected\n");
		return k->set_normalizer(new CFirstElementKernelNormalizer());
	}
	else
		SG_ERROR("Wrong kernel normalizer name.\n");

	return false;
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

bool CGUIKernel::load_kernel_init(char* filename)
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

bool CGUIKernel::save_kernel_init(char* filename)
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
	CSVM* svm=(CSVM*) ui->ui_classifier->get_classifier();
	if (svm)
	{
		if (kernel->has_property(KP_LINADD))
		{
			int32_t num_sv=svm->get_num_support_vectors();
			int32_t* sv_idx=new int32_t[num_sv];
			float64_t* sv_weight=new float64_t[num_sv];
			
			for (int32_t i=0; i<num_sv; i++)
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


bool CGUIKernel::init_kernel(char* target)
{
	if (!kernel)
		SG_ERROR("No kernel available.\n");

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

bool CGUIKernel::save_kernel(char* filename)
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

bool CGUIKernel::add_kernel(CKernel* kern, float64_t weight)
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


bool CGUIKernel::del_last_kernel()
{
	if (!kernel)
		SG_ERROR("No kernel available.\n");

	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Need a combined kernel for deleting the last kernel in it.\n");

	CKernel* last=((CCombinedKernel*) kernel)->get_last_kernel();
	if (last)
		return ((CCombinedKernel*) kernel)->delete_kernel();
	else
		SG_ERROR("No kernel available to delete.\n");

	return false;
}

bool CGUIKernel::clean_kernel()
{
	delete kernel;
	kernel=NULL;
	return true;
}

#ifdef USE_SVMLIGHT
bool CGUIKernel::resize_kernel_cache(int32_t size)
{
	if (!kernel)
		SG_ERROR("No kernel available.\n");

	kernel->resize_kernel_cache(size);
	return true;
}
#endif //USE_SVMLIGHT

bool CGUIKernel::set_optimization_type(char* opt_type)
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

bool CGUIKernel::precompute_subkernels()
{
	if (!kernel)
		SG_ERROR("No kernel available.\n");

	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Not a combined kernel.\n");

	return ((CCombinedKernel*) kernel)->precompute_subkernels();
}
#endif
