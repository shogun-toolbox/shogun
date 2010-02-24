/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _SCATTERKERNELNORMALIZER_H___
#define _SCATTERKERNELNORMALIZER_H___

#include "kernel/KernelNormalizer.h"
#include "kernel/IdentityKernelNormalizer.h"
#include "kernel/Kernel.h"

namespace shogun
{
class CScatterKernelNormalizer: public CKernelNormalizer
{

public:

	/** default constructor
	 */
	CScatterKernelNormalizer(float64_t const_diag, float64_t const_offdiag,
			CLabels* labels, CKernelNormalizer* normalizer=NULL)
	{
		m_const_diag=const_diag;
		m_const_offdiag=const_offdiag;

		SG_REF(labels);
		m_labels=labels;

		if (normalizer==NULL)
			normalizer=new CIdentityKernelNormalizer();
		SG_REF(normalizer);
		m_normalizer=normalizer;
	}

	/** default destructor */
	virtual ~CScatterKernelNormalizer()
	{
		SG_UNREF(m_labels);
		SG_UNREF(m_normalizer);
	}

	/** initialization of the normalizer
	 * @param k kernel */
	virtual bool init(CKernel* k)
	{
		m_normalizer->init(k);
		return true;
	}

	/** normalize the kernel value
	 * @param value kernel value
	 * @param idx_lhs index of left hand side vector
	 * @param idx_rhs index of right hand side vector
	 */
	inline virtual float64_t normalize(float64_t value, int32_t idx_lhs,
			int32_t idx_rhs)
	{
		value=m_normalizer->normalize(value, idx_lhs, idx_rhs);

		float64_t c=m_const_offdiag;
		if (m_labels->get_label(idx_lhs) == m_labels->get_label(idx_rhs))
			c=m_const_diag;

		return value*c;
	}

	/** normalize only the left hand side vector
	 * @param value value of a component of the left hand side feature vector
	 * @param idx_lhs index of left hand side vector
	 */
	inline virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
	{
		SG_ERROR("normalize_lhs not implemented");
		return 0;
	}

	/** normalize only the right hand side vector
	 * @param value value of a component of the right hand side feature vector
	 * @param idx_rhs index of right hand side vector
	 */
	inline virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
	{
		SG_ERROR("normalize_rhs not implemented");
		return 0;
	}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "ScatterKernelNormalizer";
	}

protected:

	/** factor to multiply to diagonal elements */
	float64_t m_const_diag;
	/** factor to multiply to off-diagonal elements */
	float64_t m_const_offdiag;

	/** labels **/
	CLabels* m_labels;

	/** labels **/
	CKernelNormalizer* m_normalizer;
};
}
#endif

