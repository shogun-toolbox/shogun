/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SVRLight_H___
#define _SVRLight_H___

#include "lib/config.h"

#ifdef USE_SVMLIGHT
#include "classifier/svm/SVM_light.h"
#endif //USE_SVMLIGHT

#ifdef USE_SVMLIGHT
/** class SVRLight */
class CSVRLight:public CSVMLight
{
	public:
		/** default constructor */
		CSVRLight();

		/** constructor
		 *
		 * @param C constant C
		 * @param epsilon epsilon
		 * @param k kernel
		 * @param lab labels
		 */
		CSVRLight(DREAL C, DREAL epsilon, CKernel* k, CLabels* lab);
		virtual ~CSVRLight() {};

		/** train regression
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifier type SVRLIGHT
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_SVRLIGHT; }

		/** SVR learn */
		void   svr_learn();

		/** compute objective function
		 *
		 * @param a a
		 * @param lin lin
		 * @param c c
		 * @param eps eps
		 * @param label label
		 * @param totdoc totdoc
		 */
		virtual double compute_objective_function(double *a, double *lin, double *c, double eps, INT *label, INT totdoc);

		/** update linear component
		 *
		 * @param docs docs
		 * @param label label
		 * @param active2dnum active2dnum
		 * @param a a
		 * @param a_old a old
		 * @param working2dnum working2dnum
		 * @param totdoc totdoc
		 * @param lin lin
		 * @param aicache ai cache
		 * @param c c
		 */
		virtual void update_linear_component(INT* docs, INT *label,
				INT *active2dnum, double *a, double* a_old,
				INT *working2dnum, INT totdoc,
				double *lin, DREAL *aicache, double* c);


		/** update linear component MKL
		 *
		 * @param docs docs
		 * @param label label
		 * @param active2dnum active2dnum
		 * @param a a
		 * @param a_old a old
		 * @param working2dnum working2dnum
		 * @param totdoc totdoc
		 * @param lin lin
		 * @param aicache ai cache
		 * @param c c
		 */
		virtual void update_linear_component_mkl(INT* docs, INT *label,
				INT *active2dnum, double *a, double* a_old,
				INT *working2dnum, INT totdoc,
				double *lin, DREAL *aicache, double* c);

		/** update linear component MKL linadd
		 *
		 * @param docs docs
		 * @param label label
		 * @param active2dnum active2dnum
		 * @param a a
		 * @param a_old a old
		 * @param working2dnum working2dnum
		 * @param totdoc totdoc
		 * @param lin lin
		 * @param aicache ai cache
		 * @param c c
		 */
		virtual void update_linear_component_mkl_linadd(INT* docs, INT *label,
				INT *active2dnum, double *a, double* a_old,
				INT *working2dnum, INT totdoc,
				double *lin, DREAL *aicache, double* c);

		/** reactivate inactive examples
		 *
		 * @param label label
		 * @param a a
		 * @param shrink_state shrink state
		 * @param lin lin
		 * @param c c
		 * @param totdoc totdoc
		 * @param iteration iteration
		 * @param inconsistent inconsistent
		 * @param docs docs
		 * @param aicache ai cache
		 * @param maxdiff maxdiff
		 */
		virtual void   reactivate_inactive_examples(INT *label,double *a,SHRINK_STATE *shrink_state,
				double *lin, double *c, INT totdoc,INT iteration,
				INT *inconsistent,
				INT *docs,DREAL *aicache,
				double* maxdiff) ;

	protected:
		/** thread helper for update linear component linadd
		 *
		 * @param params
		 */
		static void* update_linear_component_linadd_helper(void *params);

		/** regression fix index
		 *
		 * @param i i
		 * @return fix index
		 */
		inline INT regression_fix_index(INT i)
		{
			if (i>=num_vectors)
				i=2*num_vectors-1-i;

			return i;
		}

		/** regression fix index2
		 *
		 * @param i i
		 * @param num_vectors number of vectors
		 * @return fix index
		 */
		static inline INT regression_fix_index2(INT i, INT num_vectors)
		{
			if (i>=num_vectors)
				i=2*num_vectors-1-i;

			return i;
		}

		/** compute kernel at given index
		 *
		 * @param i index i
		 * @param j index j
		 * @return kernel value at i,j
		 */
		inline virtual DREAL compute_kernel(INT i, INT j)
		{
			i=regression_fix_index(i);
			j=regression_fix_index(j);

			if (use_precomputed_subkernels)
			{
				if (j>i)
					CMath::swap(i,j);
				DREAL sum=0;
				INT num_weights=-1;

				const DREAL * w = CKernelMachine::get_kernel()->get_subkernel_weights(num_weights);
				for (INT n=0; n<num_precomputed_subkernels; n++)
					if (w[n]!=0)
						sum += w[n]*precomputed_subkernels[n][i*(i+1)/2+j];
				return sum;
			}
			else
				return CKernelMachine::get_kernel()->kernel(i, j);
		}

		/** number of train elements */
		INT num_vectors;
};
#endif //USE_SVMLIGHT
#endif
