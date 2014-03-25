/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SQRTDIAGKERNELNORMALIZER_H___
#define _SQRTDIAGKERNELNORMALIZER_H___

#include <shogun/lib/config.h>
#include <shogun/kernel/normalizer/KernelNormalizer.h>
#include <shogun/kernel/string/CommWordStringKernel.h>

namespace shogun
{
/** @brief SqrtDiagKernelNormalizer divides by the Square Root of the product of
 * the diagonal elements.
 *
 * This effectively normalizes the vectors in feature space to norm 1 (see
 * CSqrtDiagKernelNormalizer)
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
		CSqrtDiagKernelNormalizer(bool use_opt_diag=false): CKernelNormalizer(),
			sqrtdiag_lhs(NULL), num_sqrtdiag_lhs(0),
			sqrtdiag_rhs(NULL), num_sqrtdiag_rhs(0),
			use_optimized_diagonal_computation(use_opt_diag)
		{
			m_parameters->add_vector(&sqrtdiag_lhs, &num_sqrtdiag_lhs, "sqrtdiag_lhs",
							  "sqrt(K(x,x)) for left hand side examples.");
			m_parameters->add_vector(&sqrtdiag_rhs, &num_sqrtdiag_rhs, "sqrtdiag_rhs",
							  "sqrt(K(x,x)) for right hand side examples.");
			SG_ADD(&use_optimized_diagonal_computation,
					"use_optimized_diagonal_computation",
					"flat if optimized diagonal computation is used", MS_NOT_AVAILABLE);
		}

		/** default destructor */
		virtual ~CSqrtDiagKernelNormalizer()
		{
			SG_FREE(sqrtdiag_lhs);
			SG_FREE(sqrtdiag_rhs);
		}

		/** initialization of the normalizer
         * @param k kernel */
		virtual bool init(CKernel* k)
		{
			ASSERT(k)
			num_sqrtdiag_lhs=k->get_num_vec_lhs();
			num_sqrtdiag_rhs=k->get_num_vec_rhs();
			ASSERT(num_sqrtdiag_lhs>0)
			ASSERT(num_sqrtdiag_rhs>0)

			CFeatures* old_lhs=k->lhs;
			CFeatures* old_rhs=k->rhs;

			k->lhs=old_lhs;
			k->rhs=old_lhs;
			bool r1=alloc_and_compute_diag(k, sqrtdiag_lhs, num_sqrtdiag_lhs);

			k->lhs=old_rhs;
			k->rhs=old_rhs;
			bool r2=alloc_and_compute_diag(k, sqrtdiag_rhs, num_sqrtdiag_rhs);

			k->lhs=old_lhs;
			k->rhs=old_rhs;

			return r1 && r2;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize(
			float64_t value, int32_t idx_lhs, int32_t idx_rhs)
		{
			float64_t sqrt_both=sqrtdiag_lhs[idx_lhs]*sqrtdiag_rhs[idx_rhs];
			return value/sqrt_both;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
		{
			return value/sqrtdiag_lhs[idx_lhs];
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
		{
			return value/sqrtdiag_rhs[idx_rhs];
		}

    public:
		/**
		 * alloc and compute the vector containing the square root of the
		 * diagonal elements of this kernel.
		 */
		bool alloc_and_compute_diag(CKernel* k, float64_t* &v, int32_t num)
		{
			SG_FREE(v);
			v=SG_MALLOC(float64_t, num);

			for (int32_t i=0; i<num; i++)
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

		/** @return object name */
		virtual const char* get_name() const { return "SqrtDiagKernelNormalizer"; }

    protected:
		/** sqrt diagonal left-hand side */
		float64_t* sqrtdiag_lhs;

		/** num sqrt diagonal left-hand side */
		int32_t num_sqrtdiag_lhs;

		/** sqrt diagonal right-hand side */
		float64_t* sqrtdiag_rhs;

		/** num sqrt diagonal right-hand side */
		int32_t num_sqrtdiag_rhs;

		/** f optimized diagonal computation is used */
		bool use_optimized_diagonal_computation;
};
}
#endif
