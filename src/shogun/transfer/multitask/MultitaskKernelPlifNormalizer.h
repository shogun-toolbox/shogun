/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Christian Widmer
 * Copyright (C) 2010 Max-Planck-Society
 */

#ifndef _MULTITASKKERNELPLIFNORMALIZER_H___
#define _MULTITASKKERNELPLIFNORMALIZER_H___

#include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
#include <shogun/kernel/Kernel.h>
#include <algorithm>
#include <vector>


namespace shogun
{
/** @brief The MultitaskKernel allows learning a piece-wise linear function (PLIF) via MKL
 *
 */
class CMultitaskKernelPlifNormalizer: public CMultitaskKernelMklNormalizer
{

public:
	/** default constructor  */
	CMultitaskKernelPlifNormalizer() : CMultitaskKernelMklNormalizer()
	{
		num_tasks = 0;
		num_tasksqr = 0;
		num_betas = 0;
	}

	/** constructor
	 */
	CMultitaskKernelPlifNormalizer(std::vector<float64_t> support_, std::vector<int32_t> task_vector)
		: CMultitaskKernelMklNormalizer()
	{

		num_betas = static_cast<int>(support_.size());

		support = support_;

		// init support points values with constant function
		betas = std::vector<float64_t>(num_betas);
		for (int i=0; i!=num_betas; i++)
		{
			betas[i] = 1;
		}

		num_tasks = get_num_unique_tasks(task_vector);
		num_tasksqr = num_tasks * num_tasks;

		// set both sides equally
		set_task_vector(task_vector);

		// init distance matrix
		distance_matrix = std::vector<float64_t>(num_tasksqr);

		// init similarity matrix
		similarity_matrix = std::vector<float64_t>(num_tasksqr);

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


	/** default destructor */
	virtual ~CMultitaskKernelPlifNormalizer()
	{
	}


	/** update cache */
	void update_cache()
	{


		for (int32_t i=0; i!=num_tasks; i++)
		{
			for (int32_t j=0; j!=num_tasks; j++)
			{

				float64_t similarity = compute_task_similarity(i, j);
				set_task_similarity(i,j,similarity);

			}

		}
	}


	/** derive similarity from distance with plif */
	float64_t compute_task_similarity(int32_t task_a, int32_t task_b)
	{

		float64_t distance = get_task_distance(task_a, task_b);
		float64_t similarity = -1;

		int32_t upper_bound_idx = -1;


		// determine interval
		for (int i=1; i!=num_betas; i++)
		{
			if (distance <= support[i])
			{
				upper_bound_idx = i;
				break;
			}
		}

		// perform interpolation (constant for beyond upper bound)
		if (upper_bound_idx == -1)
		{

			similarity = betas[num_betas-1];

		} else {

			int32_t lower_bound_idx = upper_bound_idx - 1;
			float64_t interval_size = support[upper_bound_idx] - support[lower_bound_idx];

			float64_t factor_lower = 1 - (distance - support[lower_bound_idx]) / interval_size;
			float64_t factor_upper = 1 - factor_lower;

			similarity = factor_lower*betas[lower_bound_idx] + factor_upper*betas[upper_bound_idx];

		}

		return similarity;

	}


public:

	/** @return vec task vector with containing task_id for each example on left hand side */
	virtual std::vector<int32_t> get_task_vector_lhs() const
	{
		return task_vector_lhs;
	}

	/** @param vec task vector with containing task_id for each example */
	virtual void set_task_vector_lhs(std::vector<int32_t> vec)
	{
		task_vector_lhs = vec;
	}

	/** @return vec task vector with containing task_id for each example on right hand side */
	virtual std::vector<int32_t> get_task_vector_rhs() const
	{
		return task_vector_rhs;
	}

	/** @param vec task vector with containing task_id for each example */
	virtual void set_task_vector_rhs(std::vector<int32_t> vec)
	{
		task_vector_rhs = vec;
	}

	/** @param vec task vector with containing task_id for each example */
	virtual void set_task_vector(std::vector<int32_t> vec)
	{
		task_vector_lhs = vec;
		task_vector_rhs = vec;
	}

	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @return distance between tasks
	 */
	float64_t get_task_distance(int32_t task_lhs, int32_t task_rhs)
	{

		ASSERT(task_lhs < num_tasks && task_lhs >= 0)
		ASSERT(task_rhs < num_tasks && task_rhs >= 0)

		return distance_matrix[task_lhs * num_tasks + task_rhs];

	}

	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @param distance distance between tasks
	 */
	void set_task_distance(int32_t task_lhs, int32_t task_rhs,
			float64_t distance)
	{

		ASSERT(task_lhs < num_tasks && task_lhs >= 0)
		ASSERT(task_rhs < num_tasks && task_rhs >= 0)

		distance_matrix[task_lhs * num_tasks + task_rhs] = distance;

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

	/**
	 *  @param idx index of MKL weight to get
	 */
	float64_t get_beta(int32_t idx)
	{

		return betas[idx];

	}

	/**
	 *  @param idx index of MKL weight to set
	 *  @param weight MKL weight to set
	 */
	void set_beta(int32_t idx, float64_t weight)
	{

		betas[idx] = weight;

		update_cache();

	}

	/**
	 *  @return number of kernel weights (support points)
	 */
	int32_t get_num_betas()
	{

		return num_betas;

	}


	/** @return object name */
	virtual const char* get_name() const
	{
		return "MultitaskKernelPlifNormalizer";
	}

	/** casts kernel normalizer to multitask kernel plif normalizer
	 * @param n kernel normalizer to cast
	 */
	CMultitaskKernelPlifNormalizer* KernelNormalizerToMultitaskKernelPlifNormalizer(CKernelNormalizer* n)
	{
		   return dynamic_cast<shogun::CMultitaskKernelPlifNormalizer*>(n);
	}

protected:
	/** register the parameters
	 */
	virtual void register_params()
	{
		SG_ADD(&num_tasks, "num_tasks", "the number of tasks", MS_NOT_AVAILABLE);
		SG_ADD(&num_betas, "num_betas", "the number of weights", MS_NOT_AVAILABLE);
		m_parameters->add_vector((SGString<float64_t>**)&distance_matrix, &num_tasksqr, "distance_matrix", "distance between tasks");
		m_parameters->add_vector((SGString<float64_t>**)&similarity_matrix, &num_tasksqr, "similarity_matrix", "similarity between tasks");
		m_parameters->add_vector((SGString<float64_t>**)&betas, &num_betas, "num_betas", "weights");
		m_parameters->add_vector((SGString<float64_t>**)&support, &num_betas, "support", "support points");
	}

	/** number of tasks **/
	int32_t num_tasks;
	/** square of num_tasks -- for registration purpose**/
	int32_t num_tasksqr;

	/** task vector indicating to which task each example on the left hand side belongs **/
	std::vector<int32_t> task_vector_lhs;

	/** task vector indicating to which task each example on the right hand side belongs **/
	std::vector<int32_t> task_vector_rhs;

	/** MxM matrix encoding distance between tasks **/
	std::vector<float64_t> distance_matrix;

	/** MxM matrix encoding similarity between tasks **/
	std::vector<float64_t> similarity_matrix;

	/** number of weights **/
	int32_t num_betas;

	/** weights **/
	std::vector<float64_t> betas;

	/** support points **/
	std::vector<float64_t> support;

};
}
#endif
