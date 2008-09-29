/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SQRTDIAGKERNELNORMALIZER_H___
#define _SQRTDIAGKERNELNORMALIZER_H___

#include "kernel/KernelNormalizer.h"
#include "kernel/CommWordStringKernel.h"

/** SqrtDiagKernelNormalizer divides by the Square Root of the product of
 * the diagonal elements which effectively normalizes the vectors in feature
 * space to norm 1 (see CSqrtDiagKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{k({\bf x},{\bf x'})}{\sqrt{k({\bf x},{\bf x})k({\bf x'},{\bf x'})}}
 * \f]
 */

class CSqrtDiagKernelNormalizer : public CKernelNormalizer
{
	public:
		/** default constructor
		 * @param use_opt_diag - some kernels support faster diagonal compuation
		 * via compute_diag(idx), this flag enables this
		 */
		CSqrtDiagKernelNormalizer(bool use_opt_diag=false) : sqrtdiag_lhs(NULL),
			sqrtdiag_rhs(NULL), use_optimized_diagonal_computation(use_opt_diag)
		{
		}

		/** default destructor */
		virtual ~CSqrtDiagKernelNormalizer()
		{
			delete[] sqrtdiag_lhs;
			delete[] sqrtdiag_rhs;
		}

		/** initialization of the normalizer 
         * @param k kernel */
		virtual bool init(CKernel* k)
		{
			ASSERT(k);
			INT num_lhs=k->get_num_vec_lhs();
			INT num_rhs=k->get_num_vec_rhs();
			ASSERT(num_lhs>0);
			ASSERT(num_rhs>0);

			CFeatures* old_lhs=k->lhs;
			CFeatures* old_rhs=k->rhs;

			k->lhs=old_lhs;
			k->rhs=old_lhs;
			bool r1=alloc_and_compute_diag(k, sqrtdiag_lhs, num_lhs);

			k->lhs=old_rhs;
			k->rhs=old_rhs;
			bool r2=alloc_and_compute_diag(k, sqrtdiag_rhs, num_rhs);

			k->lhs=old_lhs;
			k->rhs=old_rhs;

			return r1 && r2;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		inline virtual DREAL normalize(DREAL value, INT idx_lhs, INT idx_rhs)
		{
			DREAL sqrt_both=sqrtdiag_lhs[idx_lhs]*sqrtdiag_rhs[idx_rhs];
			return value/sqrt_both;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		inline virtual DREAL normalize_lhs(DREAL value, INT idx_lhs)
		{
			return value/sqrtdiag_lhs[idx_lhs];
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		inline virtual DREAL normalize_rhs(DREAL value, INT idx_rhs)
		{
			return value/sqrtdiag_rhs[idx_rhs];
		}

    public:
		/**
		 * alloc and compute the vector containing the square root of the
		 * diagonal elements of this kernel.
		 */
		bool alloc_and_compute_diag(CKernel* k, DREAL* &v, INT num)
		{
			delete[] v;
			v=new DREAL[num];

			for (INT i=0; i<num; i++)
			{
				if (k->get_kernel_type() == K_COMMWORDSTRING)
				{
					if (use_optimized_diagonal_computation)
						v[i]=sqrt(((CCommWordStringKernel*) k)->compute_diag(i));
					else
						v[i]=sqrt(((CCommWordStringKernel*) k)->compute_helper(i,i, true));
				}
				else
					v[i]=sqrt(k->compute(i,i));

				if (v[i]==0.0)
					v[i]=1e-16; /* avoid divide by zero exception */
			}

			return (v!=NULL);
		}

    protected:
		/** sqrt diagonal left-hand side */
		DREAL* sqrtdiag_lhs;
		/** sqrt diagonal right-hand side */
		DREAL* sqrtdiag_rhs;
		/** f optimized diagonal computation is used */
		bool use_optimized_diagonal_computation;
};

#endif
