/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _VARIANCEKERNELNORMALIZER_H___
#define _VARIANCEKERNELNORMALIZER_H___

#include <shogun/kernel/normalizer/KernelNormalizer.h>

namespace shogun
{
/** @brief VarianceKernelNormalizer divides by the ``variance''
 *
 * This effectively normalizes the vectors in feature space to variance 1 (see
 * CVarianceKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{k({\bf x},{\bf x'})}{\frac{1}{N}\sum_{i=1}^N k({\bf x}_i, {\bf x}_i) - \sum_{i,j=1}^N, k({\bf x}_i,{\bf x'}_j)/N^2}
 * \f]
 */
class CVarianceKernelNormalizer : public CKernelNormalizer
{
	public:
		/** default constructor
		 */
		CVarianceKernelNormalizer()
			: CKernelNormalizer(), meandiff(1.0), sqrt_meandiff(1.0)
		{
			SG_ADD(&meandiff, "meandiff", "Scaling constant.", MS_AVAILABLE);
			SG_ADD(&sqrt_meandiff, "sqrt_meandiff",
					"Square root of scaling constant.", MS_AVAILABLE);
		}

		/** default destructor */
		virtual ~CVarianceKernelNormalizer()
		{
		}

		/** initialization of the normalizer
         * @param k kernel */
		virtual bool init(CKernel* k)
		{
			ASSERT(k)
			int32_t n=k->get_num_vec_lhs();
			ASSERT(n>0)

			CFeatures* old_lhs=k->lhs;
			CFeatures* old_rhs=k->rhs;
			k->lhs=old_lhs;
			k->rhs=old_lhs;

			float64_t diag_mean=0;
			float64_t overall_mean=0;
			for (int32_t i=0; i<n; i++)
			{
				diag_mean+=k->compute(i, i);

				for (int32_t j=0; j<n; j++)
					overall_mean+=k->compute(i, j);
			}
			diag_mean/=n;
			overall_mean/=((float64_t) n)*n;

			k->lhs=old_lhs;
			k->rhs=old_rhs;

			meandiff=1.0/(diag_mean-overall_mean);
			sqrt_meandiff=CMath::sqrt(meandiff);

			return true;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize(
			float64_t value, int32_t idx_lhs, int32_t idx_rhs)
		{
			return value*meandiff;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
		{
			return value*sqrt_meandiff;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
		{
			return value*sqrt_meandiff;
		}

		/** @return object name */
		virtual const char* get_name() const { return "VarianceKernelNormalizer"; }

    protected:
		/** scaling constant */
		float64_t meandiff;
		/** square root of scaling constant */
		float64_t sqrt_meandiff;
};
}
#endif
