/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Christian Widmer
 * Copyright (C) 2009 Max-Planck-Society
 */

#ifndef _MULTITASKKERNELNORMALIZER_H___
#define _MULTITASKKERNELNORMALIZER_H___

#include "kernel/KernelNormalizer.h"
#include "kernel/Kernel.h"
#include <algorithm>



namespace shogun
{
/** @brief The MultitaskKernel allows Multitask Learning via a modified kernel function.
 *
 * This effectively normalizes the vectors in feature space to norm 1 (see
 * CSqrtDiagKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = \gamma(task({\bf x}),task({\bf x'})) k({\bf x},{\bf x'})
 * \f]
 */
class CMultitaskKernelNormalizer: public CKernelNormalizer
{

public:

	/** default constructor
	 */
	CMultitaskKernelNormalizer() : scale(1.0)
	{
	}

	/** default constructor
	 *
	 * @param task_lhs task vector with containing task_id for each example for left hand side
	 * @param task_rhs task vector with containing task_id for each example for right hand side
	 */
	CMultitaskKernelNormalizer(std::vector<int32_t> task_lhs, std::vector<
			int32_t> task_rhs) : scale(1.0)
	{

		set_task_vector_lhs(task_lhs);
		set_task_vector_rhs(task_rhs);

		//run sanity checks

		//invoke copy contructor
		std::vector<int32_t> unique_tasks_lhs = std::vector<int32_t>(
				task_vector_lhs.begin(), task_vector_lhs.end());
		std::vector<int32_t> unique_tasks_rhs = std::vector<int32_t>(
				task_vector_rhs.begin(), task_vector_rhs.end());

		//reorder tasks with unique prefix
		std::vector<int32_t>::iterator endLocation_lhs = std::unique(
				unique_tasks_lhs.begin(), unique_tasks_lhs.end());
		std::vector<int32_t>::iterator endLocation_rhs = std::unique(
				unique_tasks_rhs.begin(), unique_tasks_rhs.end());

		//count unique tasks
		int32_t num_unique_tasks_lhs = std::distance(unique_tasks_lhs.begin(),
				endLocation_lhs);
		int32_t num_unique_tasks_rhs = std::distance(unique_tasks_rhs.begin(),
				endLocation_rhs);

		//initialize members (lhs has always more or equally many tasks than rhs)
		num_tasks = num_unique_tasks_lhs;
		dependency_matrix = std::vector<float64_t>(num_tasks * num_tasks);

	}

	/** default destructor */
	virtual ~CMultitaskKernelNormalizer()
	{
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
			SG_INFO("using first-element normalization\n");
			scale=k->compute(0, 0);
		} else {
			SG_INFO("no inner normalization for non-WDK kernel\n");
			scale=1.0;
		}

		k->lhs=old_lhs;
		k->rhs=old_rhs;

		ASSERT(k);
		int32_t num_lhs = k->get_num_vec_lhs();
		int32_t num_rhs = k->get_num_vec_rhs();
		ASSERT(num_lhs>0);
		ASSERT(num_rhs>0);

		std::cout << "scale: " << scale << std::endl;

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

		//lookup tasks
		int32_t task_idx_lhs = task_vector_lhs[idx_lhs];
		int32_t task_idx_rhs = task_vector_rhs[idx_rhs];

		//lookup similarity
		float64_t task_similarity = get_task_similarity(task_idx_lhs,
				task_idx_rhs);

		//take task similarity into account
		float64_t similarity = (value/scale) * task_similarity;


		return similarity;

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

public:

	/** @return vec task vector with containing task_id for each example on left hand side */
	std::vector<int32_t> get_task_vector_lhs() const
	{
		return task_vector_lhs;
	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_lhs(std::vector<int32_t> vec)
	{
		task_vector_lhs = vec;
	}

	/** @return vec task vector with containing task_id for each example on right hand side */
	std::vector<int32_t> get_task_vector_rhs() const
	{
		return task_vector_rhs;
	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_rhs(std::vector<int32_t> vec)
	{
		task_vector_rhs = vec;
	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector(std::vector<int32_t> vec)
	{
		task_vector_lhs = vec;
		task_vector_rhs = vec;
	}

	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @return similarity between tasks
	 */
	float64_t get_task_similarity(int32_t task_lhs, int32_t task_rhs)
	{

		ASSERT(task_lhs < num_tasks && task_lhs >= 0);
		ASSERT(task_rhs < num_tasks && task_rhs >= 0);

		return dependency_matrix[task_lhs * num_tasks + task_rhs];

	}

	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @param similarity similarity between tasks
	 */
	void set_task_similarity(int32_t task_lhs, int32_t task_rhs,
			float64_t similarity)
	{

		ASSERT(task_lhs < num_tasks && task_lhs >= 0);
		ASSERT(task_rhs < num_tasks && task_rhs >= 0);

		dependency_matrix[task_lhs * num_tasks + task_rhs] = similarity;

	}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "MultitaskKernelNormalizer";
	}

protected:

	/** MxM matrix encoding similarity between tasks **/
	std::vector<float64_t> dependency_matrix;

	/** number of tasks **/
	int32_t num_tasks;

	/** task vector indicating to which task each example on the left hand side belongs **/
	std::vector<int32_t> task_vector_lhs;

	/** task vector indicating to which task each example on the right hand side belongs **/
	std::vector<int32_t> task_vector_rhs;

	/** scale constant obtained from k(0,0) **/
	float64_t scale;

};
}
#endif
