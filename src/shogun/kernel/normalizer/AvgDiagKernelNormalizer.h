/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _AVGDIAGKERNELNORMALIZER_H___
#define _AVGDIAGKERNELNORMALIZER_H___

#include <kernel/normalizer/KernelNormalizer.h>
namespace shogun
{
/** @brief Normalize the kernel by either a constant or the average value of the
 * diagonal elements (depending on argument c of the constructor).
 *
 * In case c <= 0 compute scale as
* \f[
* \mbox{scale} = \frac{1}{N}\sum_{i=1}^N k(x_i,x_i)
* \f]
*
* otherwise use scale=c and normalize the kernel via
*
* \f[
* k'(x,x')= \frac{k(x,x')}{scale}
* \f]
*/
class CAvgDiagKernelNormalizer : public CKernelNormalizer
{
	public:
		/** constructor
		 *
		 * @param c scale parameter, if <= 0 scaling will be computed from the
		 * avg of the kernel diagonal elements
		 */
		CAvgDiagKernelNormalizer(float64_t c=0.0) : CKernelNormalizer()
		{
			scale=c;

			SG_ADD(&scale, "scale", "Scale quotient by which kernel is scaled.",
			    MS_AVAILABLE);
		}

		/** default destructor */
		virtual ~CAvgDiagKernelNormalizer()
		{
		}

		/** initialization of the normalizer (if needed)
         * @param k kernel */
		virtual bool init(CKernel* k)
		{
			if (scale<=0)
			{
				ASSERT(k)
				int32_t num=k->get_num_vec_lhs();
				ASSERT(num>0)

				CFeatures* old_lhs=k->lhs;
				CFeatures* old_rhs=k->rhs;
				k->lhs=old_lhs;
				k->rhs=old_lhs;

				float64_t sum=0;
				for (int32_t i=0; i<num; i++)
					sum+=k->compute(i, i);

				scale=sum/num;
				k->lhs=old_lhs;
				k->rhs=old_rhs;
			}

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
			return value/scale;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
		{
			return value/sqrt(scale);
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
		{
			return value/sqrt(scale);
		}

		/** @return object name */
		virtual const char* get_name() const { return "AvgDiagKernelNormalizer"; }

	protected:
		/// the constant scaling factor (avg of diagonal or user given const)
		float64_t scale;
};
}
#endif
