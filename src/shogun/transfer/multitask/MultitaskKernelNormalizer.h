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

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>
#include <shogun/kernel/Kernel.h>
#include <algorithm>
#include <vector>



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
	CMultitaskKernelNormalizer() : CKernelNormalizer(), scale(1.0)
	{
	}

	/** default constructor
	 *
	 * @param task_vector task vector with containing task_id for each example
	 */
	CMultitaskKernelNormalizer(std::vector<int32_t> task_vector)
		: CKernelNormalizer(), scale(1.0)
	{

		num_tasks = get_num_unique_tasks(task_vector);

		// set both sides equally
		set_task_vector(task_vector);

		// init similarity matrix
		similarity_matrix = std::vector<float64_t>(num_tasks * num_tasks);

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

		if (strcmp(k->get_name(), "WeightedDegree") == 0) {
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

		//std::cout << "scale: " << scale << std::endl;

		return true;
	}

	/** helper routine
	 *
	 * @param vec vector with containing task_id for each example
	 * @return number of unique task ids
	 */
	int32_t get_num_unique_tasks(std::vector<int32_t> vec) {

		//sort
		std::sort(vec.begin(), vec.end());

		//reorder tasks with unique prefix
		std::vector<int32_t>::iterator endLocation = std::unique(vec.begin(), vec.end());

		//count unique tasks
		int32_t num_vec = std::distance(vec.begin(), endLocation);

		return num_vec;

	}

	/** normalize the kernel value
	 * @param value kernel value
	 * @param idx_lhs index of left hand side vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize(float64_t value, int32_t idx_lhs,
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

		ASSERT(task_lhs < num_tasks && task_lhs >= 0)
		ASSERT(task_rhs < num_tasks && task_rhs >= 0)

		return similarity_matrix[task_lhs * num_tasks + task_rhs];

	}

	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @param similarity similarity between tasks
	 */
	void set_task_similarity(int32_t task_lhs, int32_t task_rhs,
			float64_t similarity)
	{

		ASSERT(task_lhs < num_tasks && task_lhs >= 0)
		ASSERT(task_rhs < num_tasks && task_rhs >= 0)

		similarity_matrix[task_lhs * num_tasks + task_rhs] = similarity;

	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "MultitaskKernelNormalizer";
	}

	/** convert generic kernel normalizer object into CMultitaskKernelNormalizer
	 *
	 * @return converted CMultitaskKernelNormalizer object
	 */
	inline CMultitaskKernelNormalizer* KernelNormalizerToMultitaskKernelNormalizer(CKernelNormalizer* n)
	{
		return dynamic_cast<CMultitaskKernelNormalizer*>(n);
	}


protected:

	/** MxM matrix encoding similarity between tasks **/
	std::vector<float64_t> similarity_matrix;

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
