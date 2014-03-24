/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Christian Widmer
 * Copyright (C) 2010 Max-Planck-Society
 */

#ifndef _MULTITASKKERNELMASKPAIRNORMALIZER_H___
#define _MULTITASKKERNELMASKPAIRNORMALIZER_H___

#include <shogun/lib/config.h>
#include <shogun/kernel/normalizer/KernelNormalizer.h>
#include <shogun/kernel/Kernel.h>

#include <string>
#include <vector>
#include <utility>

namespace shogun
{


/** @brief The MultitaskKernel allows Multitask Learning via a modified kernel function.
 *
 *	Normalization is based on a mask that is defined by a number of pair of tasks.
 */
class CMultitaskKernelMaskPairNormalizer: public CKernelNormalizer
{

public:

	/** default constructor
	 */
	CMultitaskKernelMaskPairNormalizer() :
		CKernelNormalizer(), scale(1.0), normalization_constant(1.0)
	{
	}

	/** default constructor
	 * @param task_vector_
	 * @param active_pairs_
	 */
	CMultitaskKernelMaskPairNormalizer(std::vector<int32_t> task_vector_,
									   std::vector<std::pair<int32_t, int32_t> > active_pairs_) :
									   scale(1.0), normalization_constant(1.0)
	{

		set_task_vector(task_vector_);
		active_pairs = active_pairs_;

	}


	/** default destructor */
	virtual ~CMultitaskKernelMaskPairNormalizer()
	{
	}

	/** initialization of the normalizer
	 * @param k kernel */
	virtual bool init(CKernel* k)
	{
		ASSERT(k)
		int32_t num_lhs = k->get_num_vec_lhs();
		int32_t num_rhs = k->get_num_vec_rhs();
		ASSERT(num_lhs>0)
		ASSERT(num_rhs>0)


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


		return true;
	}



	/** normalize the kernel value
	 * @param value kernel value
	 * @param idx_lhs index of left hand side vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize(float64_t value, int32_t idx_lhs, int32_t idx_rhs)
	{

		//lookup tasks
		int32_t task_idx_lhs = task_vector_lhs[idx_lhs];
		int32_t task_idx_rhs = task_vector_rhs[idx_rhs];

		//lookup similarity
		float64_t task_similarity = get_similarity(task_idx_lhs, task_idx_rhs);

		//take task similarity into account
		float64_t similarity = (value/scale) * task_similarity;


		return similarity;

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

	/** @return vec task vector with containing task_id for each example on left hand side */
	std::vector<int32_t> get_task_vector_lhs() const
	{
		return task_vector_lhs;
	}


	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_lhs(std::vector<int32_t> vec)
	{

		task_vector_lhs.clear();

		for (int32_t i = 0; i != (int32_t)(vec.size()); ++i)
		{
			task_vector_lhs.push_back(vec[i]);
		}

	}

	/** @return vec task vector with containing task_id for each example on right hand side */

	std::vector<int32_t> get_task_vector_rhs() const
	{
		return task_vector_rhs;
	}


	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_rhs(std::vector<int32_t> vec)
	{

		task_vector_rhs.clear();

		for (int32_t i = 0; i != (int32_t)(vec.size()); ++i)
		{
			task_vector_rhs.push_back(vec[i]);
		}

	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector(std::vector<int32_t> vec)
	{
		set_task_vector_lhs(vec);
		set_task_vector_rhs(vec);
	}


	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @return similarity between tasks
	 */
	float64_t get_similarity(int32_t task_lhs, int32_t task_rhs)
	{

		float64_t similarity = 0.0;

		for (int32_t i=0; i!=static_cast<int>(active_pairs.size()); i++)
		{
			std::pair<int32_t, int32_t> block = active_pairs[i];

			// ignore order of pair
			if ((block.first==task_lhs && block.second==task_rhs) ||
				(block.first==task_rhs && block.second==task_lhs))
			{
				similarity = 1.0 / normalization_constant;
				break;
			}
		}


		return similarity;

	}

	/** @return vector of active pairs */
	std::vector<std::pair<int32_t, int32_t> > get_active_pairs()
	{
		return active_pairs;
	}

	/** @return normalization constant */
	float64_t get_normalization_constant () const
	{
		return normalization_constant;
	}

	/** @param constant normalization constant */
	float64_t set_normalization_constant(float64_t constant)
	{
		normalization_constant = constant;

		SG_NOTIMPLEMENTED
		return 0.0;
	}


	/** @return object name */
	virtual const char* get_name() const
	{
		return "MultitaskKernelMaskPairNormalizer";
	}

	/** casts kernel normalizer to multitask kernel mask normalizer
	 * @param n kernel normalizer to cast
	 */
	CMultitaskKernelMaskPairNormalizer* KernelNormalizerToMultitaskKernelMaskPairNormalizer(CKernelNormalizer* n)
	{
		   return dynamic_cast<shogun::CMultitaskKernelMaskPairNormalizer*>(n);
	}

protected:

	/** list of active tasks **/
	std::vector<std::pair<int32_t, int32_t> > active_pairs;

	/** task vector indicating to which task each example on the left hand side belongs **/
	std::vector<int32_t> task_vector_lhs;

	/** task vector indicating to which task each example on the right hand side belongs **/
	std::vector<int32_t> task_vector_rhs;

	/** value of first element **/
	float64_t scale;

	/** outer normalization constant **/
	float64_t normalization_constant;

};
}
#endif
