/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Christian Widmer
 * Copyright (C) 2008-2009 Max-Planck-Society
 */



#include "kernel/MultitaskKernel.h"

using namespace shogun;


MultitaskKernel::MultitaskKernel() {
	// TODO Auto-generated constructor stub

}

MultitaskKernel::MultitaskKernel(CKernel* k) {

	base_kernel = k;

}

/*
MultitaskKernel::MultitaskKernel(CKernel* k, std::vector<int32_t> task_vec_l, std::vector<int32_t> task_vec_r) {

	//base_kernel = k;
	//base_kernel->init(l, r);

	set_task_vector_lhs(task_vec_l);
	set_task_vector_rhs(task_vec_r);


	//run sanity checks

	//invoke copy contructor
	std::vector<int32_t> unique_tasks_lhs = std::vector<int32_t>(task_vector_lhs.begin(), task_vector_lhs.end());
	std::vector<int32_t> unique_tasks_rhs = std::vector<int32_t>(task_vector_rhs.begin(), task_vector_rhs.end());


	//reorder tasks with unique prefix
	std::vector<int32_t>::iterator endLocation_lhs = std::unique(unique_tasks_lhs.begin(), unique_tasks_lhs.end());
	std::vector<int32_t>::iterator endLocation_rhs = std::unique(unique_tasks_rhs.begin(), unique_tasks_rhs.end());


	//count unique tasks
	int32_t num_unique_tasks_lhs = std::distance(unique_tasks_lhs.begin(), endLocation_lhs);
	int32_t num_unique_tasks_rhs = std::distance(unique_tasks_rhs.begin(), endLocation_rhs);


	//make sure we have same number of tasks
	ASSERT(num_unique_tasks_lhs == num_unique_tasks_rhs);


	//initialize members
	this->num_tasks = num_unique_tasks_lhs;
	dependency_matrix = std::vector<float64_t>(num_tasks * num_tasks);


}
*/

MultitaskKernel::~MultitaskKernel() {
	SG_DEBUG("deleting MultitaskKernel");
}

bool MultitaskKernel::init(CFeatures* l, CFeatures* r) {

	bool is_init = base_kernel->init(lhs, rhs);
	init_normalizer();
	return is_init;

}

float64_t MultitaskKernel::compute(int32_t idx_a, int32_t idx_b) {

	//compute similarity as usual
	float64_t similarity = base_kernel->kernel(idx_a, idx_b);

	//lookup tasks
	int32_t task_idx_lhs = task_vector_lhs[idx_a];
	int32_t task_idx_rhs = task_vector_rhs[idx_b];

	//lookup similarity
	float64_t task_similarity = get_task_similarity(task_idx_lhs, task_idx_rhs);

	//std::cout << "lhs: " << task_idx_lhs << ", rhs: " << task_idx_rhs << ", sim: " << task_similarity << std::endl;

	//take task similarity into account
	similarity = similarity * task_similarity;

	return similarity;

}

