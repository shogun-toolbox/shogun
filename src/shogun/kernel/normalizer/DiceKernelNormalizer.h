/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _DICEKERNELNORMALIZER_H___
#define _DICEKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>
#include <shogun/kernel/string/CommWordStringKernel.h>

namespace shogun
{
/** @brief DiceKernelNormalizer performs kernel normalization inspired by the Dice
 * coefficient (see http://en.wikipedia.org/wiki/Dice's_coefficient)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{2k({\bf x},{\bf x'})}{k({\bf x},{\bf x})+k({\bf x'},{\bf x'}}
 * \f]
 */
class DiceKernelNormalizer : public KernelNormalizer
{
	public:
		/** default constructor
		 * @param use_opt_diag - some kernels support faster diagonal compuation
		 * via compute_diag(idx), this flag enables this
		 */
		DiceKernelNormalizer(bool use_opt_diag=false) : KernelNormalizer(),
			diag_lhs(NULL), num_diag_lhs(0), diag_rhs(NULL), num_diag_rhs(0),
			use_optimized_diagonal_computation(use_opt_diag)
		{
			/*m_parameters->add_vector(&diag_lhs, &num_diag_lhs, "diag_lhs",
							  "K(x,x) for left hand side examples.")*/;
			/*watch_param("diag_lhs", &diag_lhs, &num_diag_lhs)*/;

			/*m_parameters->add_vector(&diag_rhs, &num_diag_rhs, "diag_rhs",
							  "K(x,x) for right hand side examples.")*/;
			/*watch_param("diag_rhs", &diag_rhs, &num_diag_rhs)*/;

			/*SG_ADD(&use_optimized_diagonal_computation,
					"use_optimized_diagonal_computation",
					"flat if optimized diagonal computation is used");*/
		}

		/** default destructor */
		virtual ~DiceKernelNormalizer()
		{
			SG_FREE(diag_lhs);
			SG_FREE(diag_rhs);
		}

		/** initialization of the normalizer
         * @param k kernel */
		virtual bool init(Kernel* k)
		{
			ASSERT(k)
			num_diag_lhs=k->get_num_vec_lhs();
			num_diag_rhs=k->get_num_vec_rhs();
			ASSERT(num_diag_lhs>0)
			ASSERT(num_diag_rhs>0)

			auto old_lhs=k->lhs;
			auto old_rhs=k->rhs;

			k->lhs=old_lhs;
			k->rhs=old_lhs;
			bool r1=alloc_and_compute_diag(k, diag_lhs, num_diag_lhs);

			k->lhs=old_rhs;
			k->rhs=old_rhs;
			bool r2=alloc_and_compute_diag(k, diag_rhs, num_diag_rhs);

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
			float64_t value, int32_t idx_lhs, int32_t idx_rhs) const
		{
			float64_t diag_sum=diag_lhs[idx_lhs]*diag_rhs[idx_rhs];
			return 2*value/diag_sum;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const
		{
			SG_ERROR("linadd not supported with Dice normalization.\n")
			return 0;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const
		{
			SG_ERROR("linadd not supported with Dice normalization.\n")
			return 0;
		}

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const {
			return "DiceKernelNormalizer"; }

    protected:
		/**
		 * alloc and compute the vector containing the square root of the
		 * diagonal elements of this kernel.
		 */
		bool alloc_and_compute_diag(Kernel* k, float64_t* &v, int32_t num) const
		{
			SG_FREE(v);
			v=SG_MALLOC(float64_t, num);

			for (int32_t i=0; i<num; i++)
			{
				if (k->get_kernel_type() == K_COMMWORDSTRING)
				{
					auto cwsk = k->as<CommWordStringKernel>();
					if (use_optimized_diagonal_computation)
						v[i]=cwsk->compute_diag(i);
					else
						v[i]=cwsk->compute_helper(i,i, true);
				}
				else
					v[i]=k->compute(i,i);

				if (v[i]==0.0)
					v[i]=1e-16; /* avoid divide by zero exception */
			}

			return (v!=NULL);
		}

		/** diagonal left-hand side */
		float64_t* diag_lhs;
		/** num diag lhs */
		int32_t num_diag_lhs;

		/** diagonal right-hand side */
		float64_t* diag_rhs;
		/** num diag rhs */
		int32_t num_diag_rhs;

		/** flat if optimized diagonal computation is used */
		bool use_optimized_diagonal_computation;
};
}
#endif
