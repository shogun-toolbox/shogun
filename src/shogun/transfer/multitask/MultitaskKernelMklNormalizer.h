/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Christian Widmer
 * Copyright (C) 2010 Max-Planck-Society
 */

#ifndef _MULTITASKKERNELMKLNORMALIZER_H___
#define _MULTITASKKERNELMKLNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
#include <shogun/kernel/Kernel.h>
#include <algorithm>
#include <string>

namespace shogun
{


/** @brief Base-class for parameterized Kernel Normalizers
 *
 */
class CMultitaskKernelMklNormalizer: public CKernelNormalizer
{

public:

	/** default constructor
	 */
	CMultitaskKernelMklNormalizer() : CKernelNormalizer(), scale(1.0)
	{
		m_type = N_MULTITASK;
	}


	/** initialization of the normalizer
	 * @param k kernel */
	virtual bool init(CKernel* k)
	{

		//same as first-element normalizer
		CFeatures* old_lhs=k->lhs;
		CFeatures* old_rhs=k->rhs;
		k->lhs=old_lhs;
		k->rhs=old_lhs;

		if (std::string(k->get_name()) == "WeightedDegree") {
			SG_INFO("using first-element normalization\n")
			scale=k->compute(0, 0);
		} else {
			SG_INFO("no inner normalization for non-WDK kernel\n")
			scale=1.0;
		}


		k->lhs=old_lhs;
		k->rhs=old_rhs;

		ASSERT(k)
		int32_t num_lhs = k->get_num_vec_lhs();
		int32_t num_rhs = k->get_num_vec_rhs();
		ASSERT(num_lhs>0)
		ASSERT(num_rhs>0)


		return true;
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

public:


	/**
	 *  @param idx index of MKL weight to get
	 */
	virtual float64_t get_beta(int32_t idx) = 0;

	/**
	 *  @param idx index of MKL weight to set
	 *  @param weight MKL weight to set
	 */
	virtual void set_beta(int32_t idx, float64_t weight) = 0;


	/**
	 * @return number of sub-kernel weights for MKL
	 */
	virtual int32_t get_num_betas() = 0;


	/** @return object name */
	virtual const char* get_name() const
	{
		return "MultitaskKernelMklNormalizer";
	}

protected:


	/** scale constant obtained from k(0,0) **/
	float64_t scale;

};
}
#endif
