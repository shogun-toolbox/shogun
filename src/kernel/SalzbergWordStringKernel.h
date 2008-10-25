/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SALZBERGWORDSTRINGKERNEL_H___
#define _SALZBERGWORDSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"
#include "classifier/PluginEstimate.h"
#include "features/StringFeatures.h"

/** The SalzbergWordString kernel implements the Salzberg kernel as described in
 *
 * Engineering Support Vector Machine Kernels That Recognize Translation Initiation Sites
 * A. Zien, G.Raetsch, S. Mika, B. Schoelkopf, T. Lengauer, K.-R. Mueller
 *
 */
class CSalzbergWordStringKernel: public CStringKernel<uint16_t>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param pie the plugin estimate
		 * @param labels optional labels to set prior from
		 */
		CSalzbergWordStringKernel(INT size, CPluginEstimate* pie, CLabels* labels=NULL);

		/** constructor
	 	 *
	 	 * @param l features of left-hand side
	 	 * @param r features of right-hand side
		 * @param pie the plugin estimate
		 * @param labels optional labels to set prior from
		 */
		CSalzbergWordStringKernel(
			CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r,
			CPluginEstimate *pie, CLabels* labels=NULL);

		virtual ~CSalzbergWordStringKernel();

		/** set prior probs
		 *
		 * @param pos_prior_ positive prior
		 * @param neg_prior_ negative prior
		 */
		void set_prior_probs(DREAL pos_prior_, DREAL neg_prior_)
		{
			pos_prior=pos_prior_ ;
			neg_prior=neg_prior_ ;
			if (fabs(pos_prior+neg_prior-1)>1e-6)
				SG_WARNING( "priors don't sum to 1: %f+%f-1=%f\n", pos_prior, neg_prior, pos_prior+neg_prior-1) ;
		};

		/** set prior probs from labels
		 *
		 * @param labels labels to set prior probabilites from
		 */
		void set_prior_probs_from_labels(CLabels* labels);

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
		 * @return kernel type SALZBERG
		 */
		virtual EKernelType get_kernel_type() { return K_SALZBERG; }

		/** return the kernel's name
		 *
		 * @return name Salzberg
		 */
		virtual const char* get_name() { return "Salzberg" ; }

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
		//	DREAL compute_slow(LONG idx_a, LONG idx_b);

		/** compute index of given symbol at given position
		 *
		 * @param position position
		 * @param symbol symbol
		 * @return index
		 */
		inline INT compute_index(INT position, uint16_t symbol)
		{
			return position*num_symbols+symbol;
		}

	protected:
		/** the plugin estimate */
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

		/** number of params */
		INT num_params;
		/** number of symbols */
		INT num_symbols;
		/** sum m2 s2 */
		DREAL sum_m2_s2;
		/** positive prior */
		DREAL pos_prior;
		/** negative prior */
		DREAL neg_prior;
		/** if kernel is initialized */
		bool initialized;
};

#endif /* _SALZBERGWORDKERNEL_H__ */
