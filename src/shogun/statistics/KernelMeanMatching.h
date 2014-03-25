/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (W) 2012 Sergey Lisitsyn
 */

#ifndef KERNELMEANMATCHING_H_
#define KERNELMEANMATCHING_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

/** @brief Kernel Mean Matching */
class CKernelMeanMatching: public CSGObject
{
public:

	/** constructor */
	CKernelMeanMatching();

	/** constructor */
	CKernelMeanMatching(CKernel* kernel, SGVector<index_t> training_indices, SGVector<index_t> test_indices);

	/** get kernel */
	CKernel* get_kernel() const { SG_REF(m_kernel); return m_kernel; }
	/** set kernel */
	void set_kernel(CKernel* kernel) { SG_REF(kernel); SG_UNREF(m_kernel); m_kernel = kernel; }
	/** get training indices */
	SGVector<index_t> get_training_indices() const { return m_training_indices; }
	/** set training indices */
	void set_training_indices(SGVector<index_t> training_indices) { m_training_indices = training_indices; }
	/** get test indices */
	SGVector<index_t> get_test_indices() const { return m_test_indices; }
	/** set test indices */
	void set_test_indices(SGVector<index_t> test_indices) { m_test_indices = test_indices; }

	/** compute weights */
	SGVector<float64_t> compute_weights();

	virtual const char* get_name() const { return "KernelMeanMatching"; }

protected:

	/** kernel */
	CKernel* m_kernel;
	/** training indices */
	SGVector<index_t> m_training_indices;
	/** test indices */
	SGVector<index_t> m_test_indices;
};

}
#endif
