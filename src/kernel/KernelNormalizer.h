/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNELNORMALIZER_H___
#define _KERNELNORMALIZER_H___

class CKernel;
#include "kernel/Kernel.h"

/** The class Kernel Normalizer defines a function to postprocess kernel values, i.e.
 *
 * \f[
 * k'({\bf x},{\bf x'}) = f(k({\bf x},{\bf x'}),{\bf x},{\bf x'})
 * \f]
 *
 * examples for f(.,.,.) would be scaling with a constant
 *
 * \f[
 * f(k({\bf x},{\bf x'}), ., .)= \frac{1}{c}\cdot k({\bf x},{\bf x'})
 * \f]
 *
 * as can be found in class CAvgDiagKernelNormalizer, the identity (cf.
 * CIdentityKernelNormalizer), dividing by the Square Root of the product of
 * the diagonal elements which effectively normalizes the vectors in feature
 * space to norm 1 (see CSqrtDiagKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \frac{k({\bf x},{\bf x'})}{\sqrt{k({\bf x},{\bf x})k({\bf x'},{\bf x'})}}
 * \f]
 */
class CKernelNormalizer : public CSGObject
{
	public:
		/** default constructor */
		CKernelNormalizer() { }
		/** default destructor */
		virtual ~CKernelNormalizer() { }

		/** initialization of the normalizer (if needed)
         * @param k kernel */
		virtual bool init(CKernel* k)=0;

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual DREAL normalize(DREAL value, int32_t idx_lhs, int32_t idx_rhs)=0; 

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual DREAL normalize_lhs(DREAL value, int32_t idx_lhs)=0; 

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual DREAL normalize_rhs(DREAL value, int32_t idx_rhs)=0; 
};
#endif
