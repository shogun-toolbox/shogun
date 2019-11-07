/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _IDENTITYKERNELNORMALIZER_H___
#define _IDENTITYKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>

namespace shogun
{
/** @brief Identity Kernel Normalization, i.e. no normalization is applied. */
class IdentityKernelNormalizer : public KernelNormalizer
{
	public:
		/** default constructor */
		IdentityKernelNormalizer() : KernelNormalizer()
		{
		}

		/** default destructor */
		virtual ~IdentityKernelNormalizer()
		{
		}

		/** initialization of the normalizer (if needed)
		 * @param k kernel */
		virtual bool init(Kernel* k)
		{
			return true;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize(
				float64_t value, int32_t idx_lhs, int32_t idx_rhs) const
		{
			return value;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const
		{
			return value;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const
		{
			return value;
		}

		/** @return object name */
		virtual const char* get_name() const { return "IdentityKernelNormalizer"; }
};
}
#endif
