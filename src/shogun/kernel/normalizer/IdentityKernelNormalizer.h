/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _IDENTITYKERNELNORMALIZER_H___
#define _IDENTITYKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>

namespace shogun
{
/** @brief Identity Kernel Normalization, i.e. no normalization is applied. */
class CIdentityKernelNormalizer : public CKernelNormalizer
{
	public:
		/** default constructor */
		CIdentityKernelNormalizer() : CKernelNormalizer()
		{
		}

		/** default destructor */
		virtual ~CIdentityKernelNormalizer()
		{
		}

		/** initialization of the normalizer (if needed)
		 * @param k kernel */
		virtual bool init(CKernel* k)
		{
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
			return value;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
		{
			return value;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
		{
			return value;
		}

		/** @return object name */
		virtual const char* get_name() const { return "IdentityKernelNormalizer"; }
};
}
#endif
