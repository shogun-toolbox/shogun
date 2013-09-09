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

#include <shogun/ui/SGInterface.h>
#include <shogun/ui/GUIKernel.h>
#include <shogun/ui/GUIPluginEstimate.h>

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/AsciiFile.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/Chi2Kernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/string/LinearStringKernel.h>
#include <shogun/kernel/string/WeightedDegreeStringKernel.h>
#include <shogun/kernel/WeightedDegreeRBFKernel.h>
#include <shogun/kernel/string/SpectrumMismatchRBFKernel.h>
#include <shogun/kernel/string/WeightedDegreePositionStringKernel.h>
#include <shogun/kernel/string/FixedDegreeStringKernel.h>
#include <shogun/kernel/string/LocalityImprovedStringKernel.h>
#include <shogun/kernel/string/SimpleLocalityImprovedStringKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/ConstKernel.h>
#include <shogun/kernel/string/PolyMatchWordStringKernel.h>
#include <shogun/kernel/string/PolyMatchStringKernel.h>
#include <shogun/kernel/string/LocalAlignmentStringKernel.h>
#include <shogun/kernel/string/MatchWordStringKernel.h>
#include <shogun/kernel/string/CommWordStringKernel.h>
#include <shogun/kernel/string/WeightedCommWordStringKernel.h>
#include <shogun/kernel/string/CommUlongStringKernel.h>
#include <shogun/kernel/string/HistogramWordStringKernel.h>
#include <shogun/kernel/string/SalzbergWordStringKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/GaussianShiftKernel.h>
#include <shogun/kernel/SigmoidKernel.h>
#include <shogun/kernel/DiagKernel.h>
#include <shogun/kernel/string/OligoStringKernel.h>
#include <shogun/kernel/DistanceKernel.h>
#include <shogun/kernel/TensorProductPairKernel.h>
#include <shogun/kernel/normalizer/AvgDiagKernelNormalizer.h>
#include <shogun/kernel/normalizer/RidgeKernelNormalizer.h>
#include <shogun/kernel/normalizer/FirstElementKernelNormalizer.h>
#include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/kernel/normalizer/VarianceKernelNormalizer.h>
#include <shogun/kernel/normalizer/ScatterKernelNormalizer.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/kernel/normalizer/ZeroMeanCenterKernelNormalizer.h>
#include <shogun/kernel/WaveletKernel.h>

#include <string.h>

using namespace shogun;

CGUIKernel::CGUIKernel(CSGInterface* ui_)
: CSGObject(), ui(ui_)
{
	kernel=NULL;
}

CGUIKernel::~CGUIKernel()
{
	SG_UNREF(kernel);
}

CKernel* CGUIKernel::get_kernel()
{
	return kernel;
}

CKernel* CGUIKernel::create_oligo(int32_t size, int32_t k, float64_t width)
{
	CKernel* kern=new COligoStringKernel(size, k, width);
	SG_DEBUG("created OligoStringKernel (%p) with size %d, k %d, width %f.\n", kern, size, k, width)

	return kern;
}

CKernel* CGUIKernel::create_diag(int32_t size, float64_t diag)
{
	CKernel* kern=new CDiagKernel(size, diag);
	if (!kern)
		SG_ERROR("Couldn't create DiagKernel with size %d, diag %f.\n", size, diag)
	else
		SG_DEBUG("created DiagKernel (%p) with size %d, diag %f.\n", kern, size, diag)

	return kern;
}

CKernel* CGUIKernel::create_const(int32_t size, float64_t c)
{
	CKernel* kern=new CConstKernel(c);
	if (!kern)
		SG_ERROR("Couldn't create ConstKernel with c %f.\n", c)
	else
		SG_DEBUG("created ConstKernel (%p) with c %f.\n", kern, c)

	kern->set_cache_size(size);

	return kern;
}

CKernel* CGUIKernel::create_custom(float64_t* kmatrix, int32_t num_feat, int32_t num_vec, bool source_is_diag, bool dest_is_diag)
{
	CCustomKernel* kern=new CCustomKernel();
	SG_DEBUG("created CustomKernel (%p).\n", kern)

	SGMatrix<float64_t> km=SGMatrix<float64_t>(kmatrix, num_feat, num_vec);

	if (source_is_diag && dest_is_diag && num_feat==1)
	{
		kern->set_triangle_kernel_matrix_from_triangle(
				SGVector<float64_t>(kmatrix, num_vec));
	}
	else if (!source_is_diag && dest_is_diag && num_vec==num_feat)
		kern->set_triangle_kernel_matrix_from_full(km);
	else
		kern->set_full_kernel_matrix_from_full(km);

	return kern;
}


CKernel* CGUIKernel::create_gaussianshift(
	int32_t size, float64_t width, int32_t max_shift, int32_t shift_step)
{
	CKernel* kern=new CGaussianShiftKernel(size, width, max_shift, shift_step);
	if (!kern)
		SG_ERROR("Couldn't create GaussianShiftKernel with size %d, width %f, max_shift %d, shift_step %d.\n", size, width, max_shift, shift_step)
	else
		SG_DEBUG("created GaussianShiftKernel (%p) with size %d, width %f, max_shift %d, shift_step %d.\n", kern, size, width, max_shift, shift_step)

	return kern;
}

CKernel* CGUIKernel::create_sparsegaussian(int32_t size, float64_t width)
{
	CKernel* kern=new CGaussianKernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create GaussianKernel with size %d, width %f.\n", size, width)
	else
		SG_DEBUG("created GaussianKernel (%p) with size %d, width %f.\n", kern, size, width)

	return kern;
}

CKernel* CGUIKernel::create_gaussian(int32_t size, float64_t width)
{
	CKernel* kern=new CGaussianKernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create GaussianKernel with size %d, width %f.\n", size, width)
	else
		SG_DEBUG("created GaussianKernel (%p) with size %d, width %f.\n", kern, size, width)

	return kern;
}

CKernel* CGUIKernel::create_sigmoid(
	int32_t size, float64_t gamma, float64_t coef0)
{
	CKernel* kern=new CSigmoidKernel(size, gamma, coef0);
	if (!kern)
		SG_ERROR("Couldn't create SigmoidKernel with size %d, gamma %f, coef0 %f.\n", size, gamma, coef0)
	else
		SG_DEBUG("created SigmoidKernel (%p) with size %d, gamma %f, coef0 %f.\n", kern, size, gamma, coef0)

	return kern;
}
CKernel* CGUIKernel::create_wavelet(
	int32_t size, float64_t Wdilation, float64_t Wtranslation)
{
	CKernel* kern=new CWaveletKernel(size, Wdilation, Wtranslation);
	if (!kern)
		SG_ERROR("Couldn't create WaveletKernel with size %d, Wdilation %f, Wtranslation %f.\n", size, Wdilation, Wtranslation)
	else
		SG_DEBUG("created WaveletKernel (%p) with size %d, Wdilation %f, Wtranslation %f.\n", kern, size, Wdilation, Wtranslation)

	return kern;
}
CKernel* CGUIKernel::create_sparsepoly(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyKernel(size, degree, inhomogene);
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());
	SG_DEBUG("created PolyKernel with size %d, degree %d, inhomogene %d normalize %d.\n", kern, size, degree, inhomogene, normalize)

	return kern;
}

CKernel* CGUIKernel::create_poly(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyKernel(size, degree, inhomogene);
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());
	SG_DEBUG("created PolyKernel (%p) with size %d, degree %d, inhomogene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize)

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
		SG_ERROR("Couldn't create (Simple)LocalityImprovedStringKernel with size %d, length %d, inner_degree %d, outer_degree %d.\n", size, length, inner_degree, outer_degree)
	else
		SG_DEBUG("created (Simple)LocalityImprovedStringKernel with size %d, length %d, inner_degree %d, outer_degree %d.\n", kern, size, length, inner_degree, outer_degree)

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
		ASSERT(single_degree<order)
		for (i=0; i<order; i++)
		{
			if (i!=single_degree)
				weights[i]=0;
			else
				weights[i]=1;
		}
	}

	CKernel* kern=new CWeightedDegreeStringKernel(SGVector<float64_t>(weights, order));

	SG_DEBUG("created WeightedDegreeStringKernel (%p) with size %d, order %d, "
			"max_mismatch %d, use_normalization %d, mkl_stepsize %d, "
			"block_computation %d, single_degree %d.\n",
			kern, size, order, max_mismatch, (int) use_normalization, mkl_stepsize,
			block_computation, single_degree);

	if (!use_normalization)
		kern->set_normalizer(new CIdentityKernelNormalizer());

	((CWeightedDegreeStringKernel*) kern)->
		set_use_block_computation(block_computation);
	((CWeightedDegreeStringKernel*) kern)->set_max_mismatch(max_mismatch);
	((CWeightedDegreeStringKernel*) kern)->set_mkl_stepsize(mkl_stepsize);
	((CWeightedDegreeStringKernel*) kern)->set_which_degree(single_degree);

	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring(
	int32_t size, int32_t order, int32_t max_mismatch, int32_t length,
	int32_t center, float64_t step)
{
	int32_t i=0;
	int32_t* shifts=SG_MALLOC(int32_t, length);

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
		SG_INFO("shift[%i]=%i\n", i, shifts[i])

	float64_t* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, SGVector<float64_t>(weights, order*(1+max_mismatch)), order, max_mismatch, SGVector<int32_t>(shifts, length).clone());
	if (!kern)
		SG_ERROR("Couldn't create WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d, center %d, step %f.\n", size, order, max_mismatch, length, center, step)
	else
		SG_DEBUG("created WeightedDegreePositionStringKernel with size %d, order %d, max_mismatch %d, length %d, center %d, step %f.\n", kern, size, order, max_mismatch, length, center, step)

	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring3(
	int32_t size, int32_t order, int32_t max_mismatch, int32_t* shifts,
	int32_t length, int32_t mkl_stepsize, float64_t* position_weights)
{
	float64_t* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, SGVector<float64_t>(weights, order*(1+max_mismatch)), order, max_mismatch, SGVector<int32_t>(shifts, length, false).clone(), mkl_stepsize);
	kern->set_normalizer(new CIdentityKernelNormalizer());

	SG_DEBUG("created WeightedDegreePositionStringKernel (%p) with size %d, order %d, max_mismatch %d, length %d and position_weights (MKL stepsize: %d).\n", kern, size, order, max_mismatch, length, mkl_stepsize)

	if (!position_weights)
	{
		position_weights=SG_MALLOC(float64_t, length);
		for (int32_t i=0; i<length; i++)
			position_weights[i]=1.0/length;
	}
	((CWeightedDegreePositionStringKernel*) kern)->
		set_position_weights(SGVector<float64_t>(position_weights, length));

	return kern;
}

CKernel* CGUIKernel::create_weighteddegreepositionstring2(
	int32_t size, int32_t order, int32_t max_mismatch, int32_t* shifts,
	int32_t length, bool use_normalization)
{
	float64_t* weights=get_weights(order, max_mismatch);

	CKernel* kern=new CWeightedDegreePositionStringKernel(size, SGVector<float64_t>(weights, order*(1+max_mismatch)), order, max_mismatch, SGVector<int32_t>(shifts, length, false).clone());
	if (!use_normalization)
		kern->set_normalizer(new CIdentityKernelNormalizer());


	SG_DEBUG("created WeightedDegreePositionStringKernel (%p) with size %d, order %d, max_mismatch %d, length %d, use_normalization %d.\n", kern, size, order, max_mismatch, length, use_normalization)

	return kern;
}

float64_t* CGUIKernel::get_weights(int32_t order, int32_t max_mismatch)
{
	float64_t *weights=SG_MALLOC(float64_t, order*(1+max_mismatch));
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

CKernel* CGUIKernel::create_weighteddegreerbf(int32_t size, int32_t degree, int32_t nof_properties, float64_t width)
{
	CKernel* kern=new CWeightedDegreeRBFKernel(size, width, degree, nof_properties);
	if (!kern)
		SG_ERROR("Couldn't create WeightedDegreeRBFKernel with size %d, width %f, degree %d, nof_properties %d.\n", size, width, degree, nof_properties)
	else
		SG_DEBUG("created WeightedDegreeRBFKernel (%p) with size %d, width %f, degree %d, nof_properties %d.\n", kern, size, width, degree, nof_properties)

	return kern;
}

CKernel* CGUIKernel::create_spectrummismatchrbf(int32_t size, float64_t* AA_matrix, int32_t nr, int32_t nc, int32_t max_mismatch, int32_t degree, float64_t width)
{

  CKernel* kern = new CSpectrumMismatchRBFKernel(size, AA_matrix, nr, nc, degree, max_mismatch, width);
	if (!kern)
		SG_ERROR("Couldn't create SpectrumMismatchRBFKernel with size %d, width %f, degree %d, max_mismatch %d.\n", size, width, degree, max_mismatch)
	else
		SG_DEBUG("created SpectrumMismatchRBFKernel (%p) with size %d, width %f, degree %d, max_mismatch %d.\n", kern, size, width, degree, max_mismatch)

	return kern;

}


CKernel* CGUIKernel::create_localalignmentstring(int32_t size)
{
	CKernel* kern=new CLocalAlignmentStringKernel(size);
	if (!kern)
		SG_ERROR("Couldn't create LocalAlignmentStringKernel with size %d.\n", size)
	else
		SG_DEBUG("created LocalAlignmentStringKernel (%p) with size %d.\n", kern, size)

	return kern;
}

CKernel* CGUIKernel::create_fixeddegreestring(int32_t size, int32_t d)
{
	CKernel* kern=new CFixedDegreeStringKernel(size, d);
	if (!kern)
		SG_ERROR("Couldn't create FixedDegreeStringKernel with size %d and d %d.\n", size, d)
	else
		SG_DEBUG("created FixedDegreeStringKernel (%p) with size %d and d %d.\n", kern, size, d)

	return kern;
}

CKernel* CGUIKernel::create_chi2(int32_t size, float64_t width)
{
	CKernel* kern=new CChi2Kernel(size, width);
	if (!kern)
		SG_ERROR("Couldn't create Chi2Kernel with size %d and width %f.\n", size, width)
	else
		SG_DEBUG("created Chi2Kernel (%p) with size %d and width %f.\n", kern, size, width)

	return kern;
}

CKernel* CGUIKernel::create_commstring(
	int32_t size, bool use_sign, char* norm_str, EKernelType ktype)
{
	CKernel* kern=NULL;

	if (!norm_str)
		norm_str= (char*) "FULL";

	if (ktype==K_COMMULONGSTRING)
		kern=new CCommUlongStringKernel(size, use_sign);
	else if (ktype==K_COMMWORDSTRING)
		kern=new CCommWordStringKernel(size, use_sign);
	else if (ktype==K_WEIGHTEDCOMMWORDSTRING)
		kern=new CWeightedCommWordStringKernel(size, use_sign);

	SG_DEBUG("created WeightedCommWord/CommWord/CommUlongStringKernel (%p) with size %d, use_sign  %d norm_str %s.\n", kern, size, use_sign, norm_str)


	if (strncmp(norm_str, "NO", 2)==0)
	{
		kern->set_normalizer(new CIdentityKernelNormalizer());
	}
	else if (strncmp(norm_str, "FULL", 4)==0)
	{
		//nop, as this one is default
	}
	else
		SG_ERROR("Unsupported Normalizer requested, supports only FULL and NO\n")

	return kern;
}

CKernel* CGUIKernel::create_matchwordstring(
	int32_t size, int32_t d, bool normalize)
{
	CKernel* kern=new CMatchWordStringKernel(size, d);
	SG_DEBUG("created MatchWordStringKernel (%p) with size %d and d %d.\n", kern, size, d)
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());

	return kern;
}

CKernel* CGUIKernel::create_polymatchstring(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyMatchStringKernel(size, degree, inhomogene);
	SG_DEBUG("created PolyMatchStringKernel (%p) with size %d, degree %d, inhomogene %d normalize %d.\n", kern, size, degree, inhomogene, normalize)
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());

	return kern;
}

CKernel* CGUIKernel::create_polymatchwordstring(
	int32_t size, int32_t degree, bool inhomogene, bool normalize)
{
	CKernel* kern=new CPolyMatchWordStringKernel(size, degree, inhomogene);
	SG_DEBUG("created PolyMatchWordStringKernel (%p) with size %d, degree %d, inhomogene %d, normalize %d.\n", kern, size, degree, inhomogene, normalize)
	if (!normalize)
		kern->set_normalizer(new CIdentityKernelNormalizer());

	return kern;
}

CKernel* CGUIKernel::create_salzbergword(int32_t size)
{
	SG_INFO("Getting estimator.\n")
	CPluginEstimate* estimator=ui->ui_pluginestimate->get_estimator();
	if (!estimator)
		SG_ERROR("No estimator set.\n")

	CKernel* kern=new CSalzbergWordStringKernel(size, estimator);
	if (!kern)
		SG_ERROR("Couldn't create SalzbergWordString with size %d.\n", size)
	else
		SG_DEBUG("created SalzbergWordString (%p) with size %d.\n", kern, size)

/*
	// prior stuff
	SG_INFO("Getting labels.\n")
	CLabels* train_labels=ui->ui_labels->get_train_labels();
	if (!train_labels)
	{
		SG_INFO("Assign train labels first!\n")
		return NULL;
	}
	((CSalzbergWordStringKernel *) kern)->set_prior_probs_from_labels(train_labels);
*/

	return kern;
}

CKernel* CGUIKernel::create_histogramword(int32_t size)
{
	SG_INFO("Getting estimator.\n")
	CPluginEstimate* estimator=ui->ui_pluginestimate->get_estimator();
	if (!estimator)
		SG_ERROR("No estimator set.\n")

	CKernel* kern=new CHistogramWordStringKernel(size, estimator);
	if (!kern)
		SG_ERROR("Couldn't create HistogramWordString with size %d.\n", size)
	else
		SG_DEBUG("created HistogramWordString (%p) with size %d.\n", kern, size)

	return kern;
}

CKernel* CGUIKernel::create_linearbyte(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CLinearKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));
	SG_DEBUG("created LinearByteKernel (%p) with size %d and scale %f.\n", kern, size, scale)

	return kern;
}

CKernel* CGUIKernel::create_linearword(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CLinearKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));
	SG_DEBUG("created LinearWordKernel (%p) with size %d and scale %f.\n", kern, size, scale)

	return kern;
}

CKernel* CGUIKernel::create_linearstring(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=NULL;
	kern=new CLinearStringKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));

	SG_DEBUG("created LinearStringKernel (%p) with size %d and scale %f.\n", kern, size, scale)

	return kern;
}

CKernel* CGUIKernel::create_linear(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CLinearKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));

	SG_DEBUG("created LinearKernel (%p) with size %d and scale %f.\n", kern, size, scale)

	return kern;
}

CKernel* CGUIKernel::create_sparselinear(int32_t size, float64_t scale)
{
	size=0;
	CKernel* kern=new CLinearKernel();
	kern->set_normalizer(new CAvgDiagKernelNormalizer(scale));

	SG_DEBUG("created LinearKernel (%p) with size %d and scale %f.\n", kern, size, scale)

	return kern;
}

CKernel* CGUIKernel::create_tppk(int32_t size, float64_t* km, int32_t rows, int32_t cols)
{
	CCustomKernel* k=new CCustomKernel();
	k->set_full_kernel_matrix_from_full(SGMatrix<float64_t>(km, rows, cols));

	CKernel* kern=new CTensorProductPairKernel(size, k);

	SG_DEBUG("created TPPK (%p) with size %d and km %p, rows %d, cols %d.\n", kern, size, km, rows, cols)

	return kern;
}

CKernel* CGUIKernel::create_distance(int32_t size, float64_t width)
{
	CDistance* dist=ui->ui_distance->get_distance();
	if (!dist)
		SG_ERROR("No distance set for DistanceKernel.\n")

	CKernel* kern=new CDistanceKernel(size, width, dist);
	if (!kern)
		SG_ERROR("Couldn't create DistanceKernel with size %d and width %f.\n", size, width)
	else
		SG_DEBUG("created DistanceKernel (%p) with size %d and width %f.\n", kern, size, width)

	return kern;
}

CKernel* CGUIKernel::create_combined(
	int32_t size, bool append_subkernel_weights)
{
	CKernel* kern=new CCombinedKernel(size, append_subkernel_weights);
	if (!kern)
		SG_ERROR("Couldn't create CombinedKernel with size %d and append_subkernel_weights %d.\n", size, append_subkernel_weights)
	else
		SG_DEBUG("created CombinedKernel (%p) with size %d and append_subkernel_weights %d.\n", kern, size, append_subkernel_weights)

	return kern;
}

bool CGUIKernel::set_normalization(char* normalization, float64_t c, float64_t r)
{
	CKernel* k=kernel;

	if (k && k->get_kernel_type()==K_COMBINED)
		k=((CCombinedKernel*) kernel)->get_last_kernel();

	if (!k)
		SG_ERROR("No kernel available.\n")

	if (strncmp(normalization, "IDENTITY", 8)==0)
	{
		SG_INFO("Identity Normalization (==NO NORMALIZATION) selected\n")
		return k->set_normalizer(new CIdentityKernelNormalizer());
	}
	else if (strncmp(normalization,"AVGDIAG", 7)==0)
	{
		SG_INFO("Average Kernel Diagonal Normalization selected\n")
		return k->set_normalizer(new CAvgDiagKernelNormalizer(c));
	}
	else if (strncmp(normalization,"RIDGE", 5)==0)
	{
		SG_INFO("Ridge Kernel Normalization selected\n")
		return k->set_normalizer(new CRidgeKernelNormalizer(r, c));
	}
	else if (strncmp(normalization,"SQRTDIAG", 8)==0)
	{
		SG_INFO("Sqrt Diagonal Normalization selected\n")
		return k->set_normalizer(new CSqrtDiagKernelNormalizer());
	}
	else if (strncmp(normalization,"FIRSTELEMENT", 12)==0)
	{
		SG_INFO("First Element Normalization selected\n")
		return k->set_normalizer(new CFirstElementKernelNormalizer());
	}
	else if (strncmp(normalization,"VARIANCE", 8)==0)
	{
		SG_INFO("Variance Normalization selected\n")
		return k->set_normalizer(new CVarianceKernelNormalizer());
	}
   	else if (strncmp(normalization,"SCATTER", 7)==0)
	{
		SG_INFO("Scatter Normalization selected\n")
		CLabels* train_labels=ui->ui_labels->get_train_labels();
		ASSERT(train_labels)
		return k->set_normalizer(new CScatterKernelNormalizer(c,r, train_labels));
	}
	else if (strncmp(normalization,"ZEROMEANCENTER", 13)==0)
	{
		SG_INFO("Zero Mean Center Normalization selected\n")
		return k->set_normalizer(new CZeroMeanCenterKernelNormalizer());
	}
	else
		SG_ERROR("Wrong kernel normalizer name.\n")

	SG_UNREF(k);

	return false;
}

bool CGUIKernel::set_kernel(CKernel* kern)
{
	if (kern)
	{
		SG_DEBUG("deleting old kernel (%p).\n", kernel)
		SG_REF(kern);
		SG_UNREF(kernel);
		kernel=kern;
		SG_DEBUG("set new kernel (%p).\n", kern)

		return true;
	}
	else
		return false;
}

bool CGUIKernel::init_kernel_optimization()
{
	CSVM* svm=(CSVM*) ui->ui_classifier->get_classifier();
	if (svm)
	{
		if (kernel->has_property(KP_LINADD))
		{
			int32_t num_sv=svm->get_num_support_vectors();
			int32_t* sv_idx=SG_MALLOC(int32_t, num_sv);
			float64_t* sv_weight=SG_MALLOC(float64_t, num_sv);

			for (int32_t i=0; i<num_sv; i++)
			{
				sv_idx[i]=svm->get_support_vector(i);
				sv_weight[i]=svm->get_alpha(i);
			}

			bool ret=kernel->init_optimization(num_sv, sv_idx, sv_weight);

			SG_FREE(sv_idx);
			SG_FREE(sv_weight);

			if (!ret)
				SG_ERROR("Initialization of kernel optimization failed\n")
			return ret;
		}
	}
	else
		SG_ERROR("Create SVM first!\n")

	return true;
}

bool CGUIKernel::delete_kernel_optimization()
{
	if (kernel && kernel->has_property(KP_LINADD) && kernel->get_is_initialized())
		kernel->delete_optimization();

	return true;
}


bool CGUIKernel::init_kernel(const char* target)
{
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	// no need to init custom kernel
	if (kernel->get_kernel_type() == K_CUSTOM || !target)
	{
		initialized=true;
		return true;
	}

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
				SG_INFO("Initialising kernel with TRAIN DATA, train: %p\n", train)
				kernel->init(train, train);
				initialized=true;
			}
			else
				SG_ERROR("Kernel can not process this train feature type: %d %d.\n", fclass, ftype)
		}
		else
			SG_DEBUG("Not initing kernel - no train features assigned.\n")
	}
	else if (!strncmp(target, "TEST", 4))
	{
		CFeatures* train=ui->ui_features->get_train_features();
		CFeatures* test=ui->ui_features->get_test_features();
		if (train && test)
		{
			EFeatureClass fclass=test->get_feature_class();
			EFeatureType ftype=test->get_feature_type();
			if ((k_fclass==fclass || k_fclass==C_ANY || fclass==C_ANY) &&
				(k_ftype==ftype || k_ftype==F_ANY || ftype==F_ANY))

			{
				if (!initialized)
				{
					EFeatureClass tr_fclass=train->get_feature_class();
					EFeatureType tr_ftype=train->get_feature_type();
					if ((k_fclass==tr_fclass || k_fclass==C_ANY || tr_fclass==C_ANY) &&
							(k_ftype==tr_ftype || k_ftype==F_ANY || tr_ftype==F_ANY))
					{
						SG_INFO("Initialising kernel with TRAIN DATA, train: %p\n", train)
						kernel->init(train, train);
						initialized=true;
					}
					else
						SG_ERROR("Kernel can not process this train feature type: %d %d.\n", fclass, ftype)
				}

				SG_INFO("Initialising kernel with TEST DATA, train: %p test %p\n", train, test)
				// lhs -> always train_features; rhs -> always test_features
				kernel->init(train, test);
			}
			else
				SG_ERROR("Kernel can not process this test feature type: %d %d.\n", fclass, ftype)
		}
		else
			SG_DEBUG("Not initing kernel - no train and test features assigned.\n")
	}
	else
		SG_ERROR("Unknown target %s.\n", target)

	return true;
}

bool CGUIKernel::save_kernel(char* filename)
{
	if (kernel && initialized)
	{
		CAsciiFile* file=new CAsciiFile(filename);
		try
		{
			kernel->save(file);
		}
		catch (...)
		{
			SG_ERROR("Writing to file %s failed!\n", filename)
		}

		SG_UNREF(file);
		SG_INFO("Successfully written kernel to \"%s\" !\n", filename)
		return true;
	}
	else
		SG_ERROR("No kernel set / kernel not initialized!\n")

	return false;
}

bool CGUIKernel::add_kernel(CKernel* kern, float64_t weight)
{
	if (!kern)
		SG_ERROR("Given kernel to add is invalid.\n")

	if (!kernel)
	{
		kernel= new CCombinedKernel(20, false);
		SG_REF(kernel);
	}

	if (kernel->get_kernel_type()!=K_COMBINED)
	{
		CKernel* first_elem=kernel;
		kernel= new CCombinedKernel(20, false);
		SG_REF(kernel);
		((CCombinedKernel*) kernel)->append_kernel(first_elem);
	}

	if (!kernel)
		SG_ERROR("Combined kernel object could not be created.\n")

	kern->set_combined_kernel_weight(weight);

	bool success=((CCombinedKernel*) kernel)->append_kernel(kern);

	initialized=true;
	if (success)
		((CCombinedKernel*) kernel)->list_kernels();
	else
		SG_ERROR("Adding of kernel failed.\n")

	return success;
}


bool CGUIKernel::del_last_kernel()
{
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Need a combined kernel for deleting the last kernel in it.\n")

	if (((CCombinedKernel*) kernel)->get_num_kernels()>0)
		return ((CCombinedKernel*) kernel)->
				delete_kernel(((CCombinedKernel*) kernel)->get_num_kernels()-1);
	else
		SG_ERROR("No kernel available to delete.\n")

	return false;
}

bool CGUIKernel::clean_kernel()
{
	SG_UNREF(kernel);
	kernel=NULL;
	return true;
}

#ifdef USE_SVMLIGHT
bool CGUIKernel::resize_kernel_cache(int32_t size)
{
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	kernel->resize_kernel_cache(size);
	return true;
}
#endif //USE_SVMLIGHT

bool CGUIKernel::set_optimization_type(char* opt_type)
{
	EOptimizationType opt=SLOWBUTMEMEFFICIENT;
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	if (strncmp(opt_type, "FASTBUTMEMHUNGRY", 16)==0)
	{
		SG_INFO("FAST METHOD selected\n")
		opt=FASTBUTMEMHUNGRY;
		kernel->set_optimization_type(opt);

		return true;
	}
	else if (strncmp(opt_type,"SLOWBUTMEMEFFICIENT", 19)==0)
	{
		SG_INFO("MEMORY EFFICIENT METHOD selected\n")
		opt=SLOWBUTMEMEFFICIENT;
		kernel->set_optimization_type(opt);

		return true;
	}
	else
		SG_ERROR("Wrong kernel optimization type.\n")

	return false;
}

bool CGUIKernel::precompute_subkernels()
{
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	if (kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("Not a combined kernel.\n")

	return ((CCombinedKernel*) kernel)->precompute_subkernels();
}
