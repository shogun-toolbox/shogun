/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Jonas Behr
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __KERNELPARAM_h__
#define __KERNELPARAM_h__

#include "lib/common.h"
#include "base/SGObject.h"

#include <ui/GUIKernel.h>
#include <shogun/kernel/WeightedDegreePositionStringKernel.h>
#include <shogun/kernel/WeightedDegreeStringKernel.h>
#include <shogun/kernel/CommWordStringKernel.h>
#include <shogun/kernel/WeightedCommWordStringKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/SparseLinearKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/SalzbergWordStringKernel.h>
#include <shogun/features/SimpleFeatures.h>
//#include <shogun/features/PolyFeatures.h>
#include <shogun/preproc/SortWordString.h>

/** @brief class KernelParam */
class CKernelParam : public CSGObject 
{
	public:

		/** constructor
		 */
		CKernelParam();

		virtual ~CKernelParam();

		/**
		 *
		 */
		CKernel* create_kernel(CGUIKernel* ui_kernel);

		void set_order(int32_t p_order){ order=p_order;};
		void set_kernelname(char* p_kernelname)
		{ 
			delete[] kernelname;
			kernelname=p_kernelname;
		};

		/** 
		 * @return object name 
		 */
		inline virtual const char* get_name() const { return "KernelParam"; }
	protected:
		float64_t* get_weights();
                int32_t size;
		int32_t order;
                int32_t max_mismatch;
		bool use_normalization;
                int32_t mkl_stepsize;
                bool block_computation;
                int32_t single_degree;
		int32_t length;
		int32_t center;
		int32_t step;
		char* kernelname;

};
#endif
