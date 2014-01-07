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

#include <kernel/normalizer/KernelNormalizer.h>
#include <kernel/normalizer/IdentityKernelNormalizer.h>
#include <kernel/Kernel.h>
#include <labels/Labels.h>
#include <labels/MulticlassLabels.h>
#include <io/SGIO.h>

namespace shogun
{
/** @brief the scatter kernel normalizer */
class CScatterKernelNormalizer: public CKernelNormalizer
{

public:
	/** default constructor  */
	CScatterKernelNormalizer() : CKernelNormalizer()
	{
		init();
	}

	/** default constructor
	 */
	CScatterKernelNormalizer(float64_t const_diag, float64_t const_offdiag,
			CLabels* labels,CKernelNormalizer* normalizer=NULL)
		: CKernelNormalizer()
	{
		init();

		m_testing_class=-1;
		m_const_diag=const_diag;
		m_const_offdiag=const_offdiag;

		ASSERT(labels)
		SG_REF(labels);
		m_labels=labels;
		ASSERT(labels->get_label_type()==LT_MULTICLASS)
		labels->ensure_valid();

		if (normalizer==NULL)
			normalizer=new CIdentityKernelNormalizer();
		SG_REF(normalizer);
		m_normalizer=normalizer;

		SG_DEBUG("Constructing ScatterKernelNormalizer with const_diag=%g"
				" const_offdiag=%g num_labels=%d and normalizer='%s'\n",
				const_diag, const_offdiag, labels->get_num_labels(),
				normalizer->get_name());
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

	/** get testing class
	 *
	 * @return testing class (-1 disabled, 0...class otherwise)
	 */
	int32_t get_testing_class()
	{
		return m_testing_class;
	}

	/** set testing status
	 *
	 * @param c set class to test for
	 */
	void set_testing_class(int32_t c)
	{
		m_testing_class=c;
	}

	/** normalize the kernel value
	 * @param value kernel value
	 * @param idx_lhs index of left hand side vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize(float64_t value, int32_t idx_lhs,
			int32_t idx_rhs)
	{
		value=m_normalizer->normalize(value, idx_lhs, idx_rhs);
		float64_t c=m_const_offdiag;

		if (m_testing_class>=0)
		{
			if (((CMulticlassLabels*) m_labels)->get_label(idx_lhs) == m_testing_class)
				c=m_const_diag;
		}
		else
		{
			if (((CMulticlassLabels*) m_labels)->get_label(idx_lhs) == ((CMulticlassLabels*) m_labels)->get_label(idx_rhs))
				c=m_const_diag;

		}
		return value*c;
	}

	/** normalize only the left hand side vector
	 * @param value value of a component of the left hand side feature vector
	 * @param idx_lhs index of left hand side vector
	 */
	virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
	{
		SG_ERROR("normalize_lhs not implemented")
		return 0;
	}

	/** normalize only the right hand side vector
	 * @param value value of a component of the right hand side feature vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
	{
		SG_ERROR("normalize_rhs not implemented")
		return 0;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "ScatterKernelNormalizer";
	}

private:
	void init()
	{
		m_const_diag = 1.0;
		m_const_offdiag = 1.0;

		m_labels = NULL;
		m_normalizer = NULL;

		m_testing_class = -1;

		SG_ADD(&m_testing_class, "m_testing_class",
				"Testing Class.", MS_NOT_AVAILABLE);
		SG_ADD(&m_const_diag, "m_const_diag",
				"Factor to multiply to diagonal elements.", MS_AVAILABLE);
		SG_ADD(&m_const_offdiag, "m_const_offdiag",
				"Factor to multiply to off-diagonal elements.", MS_AVAILABLE);

		SG_ADD((CSGObject**) &m_labels, "m_labels", "Labels", MS_NOT_AVAILABLE);
		SG_ADD((CSGObject**) &m_normalizer, "m_normalizer", "Kernel normalizer.",
		    MS_AVAILABLE);
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

	/** upon testing which class to test for */
	int32_t m_testing_class;
};
}
#endif

