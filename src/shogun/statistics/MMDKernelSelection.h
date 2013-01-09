/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTION_H_
#define __MMDKERNELSELECTION_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

class CKernelTwoSampleTestStatistic;
class CKernel;

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

	/** TODO */
	virtual SGVector<float64_t> compute_measures()=0;

	/** TODO */
	virtual CKernel* select_kernel();

	/** @return name of the SGSerializable */
	const char* get_name() const=0;

protected:
//	/** Polymorphic function that computes the MMD kernel choice criterion for
//	 * the provided kernel. This criterion is always assumed to have to be
//	 * maximised, so in case of type II error related measures, add a minus
//	 * when implementing in subclasses.
//	 *
//	 * @return kernel criterion for given kernel, has to be maximised.
//	 * @param kernel kernel to compute the measure for
//	 */
//	virtual float64_t compute_measure(CKernel* kernel)=0;

private:

	/** Initializer */
	void init();

protected:
	/** Underlying MMD instance */
	CKernelTwoSampleTestStatistic* m_mmd;
};

}

#endif /* __MMDKERNELSELECTION_H_ */
