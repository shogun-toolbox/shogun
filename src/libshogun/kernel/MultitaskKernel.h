/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Christian Widmer
 * Copyright (C) 2008-2009 Max-Planck-Society
 */


#ifndef MULTITASKKERNEL_H_
#define MULTITASKKERNEL_H_


#include "kernel/Kernel.h"
#include "base/SGObject.h"
#include "lib/common.h"
#include "features/Features.h"

#include <algorithm>

namespace shogun {


/** @brief The MultitaskKernel allows Multitask Learning via a modified kernel function.
 *
 * more to come...
 *
 */
class MultitaskKernel: public CKernel {

public:


	MultitaskKernel();
	MultitaskKernel(CKernel*);
	MultitaskKernel(CKernel*, std::vector<int32_t> task_vec_l, std::vector<int32_t> task_vec_r);

	virtual ~MultitaskKernel();


	/** initialize kernel
	 *  e.g. setup lhs/rhs of kernel, precompute normalization constants etc.
	 *  make sure to check that your kernel can deal with the
	 *  supplied features (!)
	 */
	virtual bool init(CFeatures* lhs, CFeatures* rhs);


	std::vector<int32_t> get_task_vector_lhs() const {
		return task_vector_lhs;
	}

	void set_task_vector_lhs(std::vector<int32_t> vec) {
		task_vector_lhs = vec;
	}

	std::vector<int32_t> get_task_vector_rhs() const {
		return task_vector_rhs;
	}

	void set_task_vector_rhs(std::vector<int32_t> vec) {
		task_vector_rhs = vec;
	}

	void set_task_vector(std::vector<int32_t> vec) {
		task_vector_lhs = vec;
		task_vector_rhs = vec;
	}


	float64_t get_task_similarity(int32_t task1, int32_t task2) {

		ASSERT(task1 < num_tasks && task1 >= 0);
		ASSERT(task2 < num_tasks && task2 >= 0);

		return dependency_matrix[task1 * num_tasks + task2];

	}

	void set_task_similarity(int32_t task1, int32_t task2, float64_t similarity) {

		ASSERT(task1 < num_tasks && task1 >= 0);
		ASSERT(task2 < num_tasks && task2 >= 0);

		dependency_matrix[task1 * num_tasks + task2] = similarity;

	}


	virtual float64_t compute(int32_t idx_a, int32_t idx_b);


	/** return the kernel's name
	 *
	 * @return name Custom
	 */
	virtual const char* get_name() const { return "MultitaskKernel"; }


    /** return feature type the kernel can deal with
     *
     * @return feature type
     **/

    virtual EFeatureType get_feature_type() {
    	if (base_kernel) {
    		return base_kernel->get_feature_type();
    	} else {
    		return F_UNKNOWN;
    	}
    }

    /* return feature class the kernel can deal with
     *
     * @return feature class
     */
    virtual EFeatureClass get_feature_class() {
    	if (base_kernel) {
    		return base_kernel->get_feature_class();
    	} else {
    		return C_UNKNOWN;
    	}

    }


protected:

	CKernel* base_kernel;

	std::vector<float64_t> dependency_matrix;

	int32_t num_tasks;

	std::vector<int32_t> task_vector_lhs;
	std::vector<int32_t> task_vector_rhs;

};

}

#endif /* MULTITASKKERNEL_H_ */
