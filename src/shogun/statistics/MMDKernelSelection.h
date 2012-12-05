/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTION_H_
#define __MMDKERNELSELECTION_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

class CList;
class CKernel;
class CKernelTwoSampleTestStatistic;

class CMMDKernelSelection: public CSGObject
{
public:

	/** Default constructor */
	CMMDKernelSelection();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test. Currently: linear or quadratic time MMD.
	 */
	CMMDKernelSelection(CKernelTwoSampleTestStatistic* mmd);

	/** Destructor */
	virtual ~CMMDKernelSelection();

	/** Computes the underlying kernel criterion on all attached kernels and
	 * returns a vector with these values
	 *
	 * @return vector with kernel criterion values for all attached kernels
	 */
	virtual SGVector<float64_t> compute_measures();

	/** Computes the underlying kernel criterion on all attached kernels and
	 * returns the one with the maximum value.
	 *
	 * @return kernel that lead to the largest kernel criterion value. Is
	 * SG_REF'ed.
	 */
	virtual CKernel* select_kernel();

	/** Adds a kernel to the list of to be considered kernels
	 * @param kernel is added to the internal list. Is SG_REF'ed.
	 */
	virtual void add_kernel(CKernel* kernel);

	/** Removes all kernel from the list of to be considered kernels */
	virtual void remove_all_kernels();

	/** @return name of the SGSerializable */
	const char* get_name() const=0;

protected:
	/** Polymorphic function that computes the MMD kernel choice criterion for
	 * the provided kernel. This criterion is always assumed to have to be
	 * maximised, so in case of type II error related measures, add a minus
	 * when implementing in subclasses.
	 *
	 * @param kernel kernel to compute criterion for
	 * @return kernel criterion for given kernel, has to be maximised.
	 */
	virtual float64_t compute_measure(CKernel* kernel)=0;

private:

	/** Initializer */
	void init();

protected:
	/** List of to be considered kernels */
	CList* m_kernel_list;

	/** Underlying MMD instance */
	CKernelTwoSampleTestStatistic* m_mmd;
};

}

#endif /* __MMDKERNELSELECTION_H_ */
