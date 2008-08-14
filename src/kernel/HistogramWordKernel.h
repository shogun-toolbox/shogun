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

#ifndef _HISTOGRAMWORDKERNEL_H___
#define _HISTOGRAMWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"
#include "classifier/PluginEstimate.h"
#include "features/StringFeatures.h"

/** The HistogramWord computes the TOP kernel on inhomogeneous Markov Chains. */
class CHistogramWordKernel: public CStringKernel<WORD>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param pie plugin estimate
		 */
		CHistogramWordKernel(INT size, CPluginEstimate* pie);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param pie plugin estimate
		 */
		CHistogramWordKernel(
			CStringFeatures<WORD>* l, CStringFeatures<WORD>* r,
			CPluginEstimate* pie);

		virtual ~CHistogramWordKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** load kernel init_data
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		bool load_init(FILE* src);

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		bool save_init(FILE* dest);

		/** return what type of kernel we are
		 *
		 * @return kernel type HISTOGRAM
		 */
		virtual EKernelType get_kernel_type() { return K_HISTOGRAM; }

		/** return the kernel's name
		 *
		 * @return name Histogram
		 */
		virtual const CHAR* get_name() { return "Histogram" ; } ;

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		DREAL compute(INT idx_a, INT idx_b);

		/** compute index
		 *
		 * @param position position
		 * @param symbol symbol
		 * @return index at given position in given symbol
		 */
		inline INT compute_index(INT position, WORD symbol)
		{
			return position*num_symbols+symbol+1;
		}

	protected:
		/** plugin estimate */
		CPluginEstimate* estimate;

		/** mean */
		DREAL* mean;
		/** variance */
		DREAL* variance;

		/** sqrt diagonal of left-hand side */
		DREAL* sqrtdiag_lhs;
		/** sqrt diagonal of right-hand side */
		DREAL* sqrtdiag_rhs;

		/** ld mean left-hand side */
		DREAL* ld_mean_lhs;
		/** ld mean right-hand side */
		DREAL* ld_mean_rhs;

		/** plo left-hand side */
		DREAL* plo_lhs;
		/** plo right-hand side */
		DREAL* plo_rhs;

		/** number of parameters */
		INT num_params;
		/** number of parameters2 */
		INT num_params2;
		/** number of symbols */
		INT num_symbols;
		/** sum m2 s2 */
		DREAL sum_m2_s2;

		/** if kernel is initialized */
		bool initialized;
};

#endif /* _HISTOGRAMWORDKERNEL_H__ */
