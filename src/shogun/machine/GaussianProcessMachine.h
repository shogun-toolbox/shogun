/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#ifndef _GAUSSIANPROCESSMACHINE_H_
#define _GAUSSIANPROCESSMACHINE_H_

#include <shogun/lib/config.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/gp/InferenceMethod.h>

#ifdef HAVE_EIGEN3

namespace shogun
{

/** @brief A base class for Gaussian Processes.
 *
 * Instead of a distribution over weights, the GP specifies a
 * distribution over functions:
 *
 * \f[
 * f(x) \sim GP(m(x), k(x,x'))
 * \f]
 *
 * where \f$m(x)\f$ - mean function, \f$k(x, x')\f$ - covariance function.
 */
class CGaussianProcessMachine : public CMachine
{
public:
	/** default constructor */
	CGaussianProcessMachine();

	/** constructor
	 *
	 * @param method inference method
	 */
	CGaussianProcessMachine(CInferenceMethod* method);

	virtual ~CGaussianProcessMachine();

	/** return name of the machine
	 *
	 * @return name GaussianProcessMachine
	 */
	virtual const char* get_name() const { return "GaussianProcessMachine"; }

	/** get inference method
	 *
	 * @return inference method, which is used by Gaussian process machine
	 */
	CInferenceMethod* get_inference_method() const
	{
		SG_REF(m_method);
		return m_method;
	}

	/** set inference method
	 *
	 * @param method inference method
	 */
	void set_inference_method(CInferenceMethod* method)
	{
		SG_REF(method);
		SG_UNREF(m_method);
		m_method=method;
	}

	/** set training labels
	 *
	 * @param lab labels to set
	 */
	virtual void set_labels(CLabels* lab)
	{
		CMachine::set_labels(lab);
		m_method->set_labels(lab);
	}

private:
	void init();

protected:
	/** inference method */
	CInferenceMethod* m_method;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _GAUSSIANPROCESSMACHINE_H_ */
