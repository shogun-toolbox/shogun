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

#ifndef __GUIKERNEL__H
#define __GUIKERNEL__H

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "base/SGObject.h"
#include "kernel/Kernel.h"

class CSGInterface;

class CGUIKernel : public CSGObject
{
 public:
 	/** constructor */
	CGUIKernel(CSGInterface* interface);

	/** destructor */
	~CGUIKernel();

	/** get active kernel */
	CKernel* get_kernel();
	/** set active kernel */
	bool set_kernel(CKernel* kern);
	/** add kernel to a Combined kernel, creating one if necessary */
	bool add_kernel(CKernel* kern, DREAL weight=1);
	/** delete last kernel in combined kernel */
	bool del_last_kernel();

	/** initialize kernel */
	bool init_kernel(CHAR* target);
	/** initialize kernel  optimization */
	bool init_kernel_optimization();
	/** delete kernel optimization */
	bool delete_kernel_optimization();
	/** load kernel initialization from file */
	bool load_kernel_init(CHAR* filename);
	/** save kernel initialization to file */
	bool save_kernel_init(CHAR* filename);
	/** save kernel (matrix) to file */
	bool save_kernel(CHAR* filename);
	/** clean/r kernel */
	bool clean_kernel();
#ifdef USE_SVMLIGHT
	/** resize kernel cache */
	bool resize_kernel_cache(INT size);
#endif //USE_SVMLIGHT
	/** set optimization type */
	bool set_optimization_type(CHAR* opt_type);

	/** check if kernel is initialized */
	bool is_initialized() { return initialized; }

#ifdef HAVE_MINDY
	/** create a new MindyGram kernel */
	CKernel* CGUIKernel::create_mindygram(
		INT size=10, CHAR* meas_str=NULL, CHAR* norm_str=NULL,
		DREAL width=1, CHAR* param_str=NULL)
#endif
	/** create a new Diag kernel */
	CKernel* create_diag(INT size=10, DREAL diag=1);
	/** create a new Const kernel */
	CKernel* create_const(INT size=10, DREAL c=1);
	/** create a new Custom kernel */
	CKernel* create_custom();
	/** create a new GaussianShift kernel */
	CKernel* create_gaussianshift(
		INT size=10, DREAL width=1, INT max_shift=0, INT shift_step=1);
	/** create a new SparseGaussian kernel */
	CKernel* create_sparsegaussian(INT size=10, DREAL width=1);
	/** create a new Gaussian kernel */
	CKernel* create_gaussian(INT size=10, DREAL width=1);
	/** create a new Sigmoid kernel */
	CKernel* create_sigmoid(INT size=10, DREAL gamma=0.01, DREAL coef0=0);
	/** create a new SparsePoly kernel */
	CKernel* create_sparsepoly(
		INT size=10, INT degree=2, bool inhomogene=false,
		bool normalize=true);
	/** create a new Poly kernel */
	CKernel* create_poly(
		INT size=10, INT degree=2, bool inhomogene=false,
		bool normalize=true);
	/** create a new (Simple)LocalityImprovedString kernel */
	CKernel* create_localityimprovedstring(
		INT size=10, INT length=3, INT inner_degree=3,
		INT outer_degree=1, EKernelType ktype=K_LOCALITYIMPROVED);
	/** create a new WeightedDegreeString kernel */
	CKernel* create_weighteddegreestring(
		INT size=10, INT order=3, INT max_mismatch=1,
		bool use_normalization=true, INT mkl_stepsize=1,
		bool block_computation=true, INT single_degree=-1);
	/** create a new WeightedDegreePositionString kernel */
	CKernel* create_weighteddegreepositionstring(
		INT size=10, INT order=3, INT max_mismatch=1, INT length=0,
		INT center=0, DREAL step=1);
	CKernel* create_weighteddegreepositionstring3(
		INT size=10, INT order=3, INT max_mismatch=1,
		INT* shifts=NULL, INT length=0, INT mkl_stepsize=1,
		DREAL* position_weights=NULL);
	CKernel* create_weighteddegreepositionstring2(
		INT size=10, INT order=3, INT max_mismatch=1,
		INT* shifts=NULL, INT length=0, bool use_normalization=true);
	/** create a new LocalAlignmentString kernel */
	CKernel* create_localalignmentstring(INT size=10);
	/** create a new FixedDegreeString kernel */
	CKernel* create_fixeddegreestring(INT size=10, INT d=3);
	/** create a new Chi2 kernel */
	CKernel* create_chi2(INT size=10, DREAL width=1);
	/** create a new WeightedCommWord/CommWord/CommULongString kernel */
	CKernel* create_commstring(
		INT size=10, bool use_sign=false, CHAR* norm_str=NULL,
		EKernelType ktype=K_WEIGHTEDCOMMWORDSTRING);
	/** create a new WordMatch kernel */
	CKernel* create_wordmatch(INT size=10, INT d=3);
	/** create a new PolyMatchString kernel */
	CKernel* create_polymatchstring(
		INT size=10, INT degree=2, bool inhomogene=false, bool normalize=true);
	/** create a new PolyMatchWord kernel */
	CKernel* create_polymatchword(
		INT size=10, INT degree=2, bool inhomogene=false, bool normalize=true);
	/** create a new SalzbergWord kernel */
	CKernel* create_salzbergword(INT size=10);
	/** create a new HistogramWord kernel */
	CKernel* create_histogramword(INT size=10);
	/** create a new LinearByte kernel */
	CKernel* create_linearbyte(INT size=10, DREAL scale=-1);
	/** create a new LinearWord kernel */
	CKernel* create_linearword(INT size=10, DREAL scale=-1);
	/** create a new LinearString kernel */
	CKernel* create_linearstring(INT size=10, DREAL scale=-1);
	/** create a new Linear kernel */
	CKernel* create_linear(INT size=10, DREAL scale=-1);
	/** create a new SparseLinear kernel */
	CKernel* create_sparselinear(INT size=10, DREAL scale=-1);
	/** create a new Distance kernel */
	CKernel* create_distance(INT size=10, DREAL width=1);
	/** create a new Combined kernel */
	CKernel* create_combined(
		INT size=10, bool append_subkernel_weights=false);


 protected:
	CKernel* kernel;
	CSGInterface* ui;
	bool initialized;

 private:
	ENormalizationType get_normalization_from_str(CHAR* str);
	DREAL* get_weights(INT order, INT max_mismatch);

};
#endif //HAVE_SWIG
#endif
