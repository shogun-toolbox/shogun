/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Roman Votyakov
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 * Code adapted from
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and 
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
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
 * Instead of a distribution over weights, the GP specifies a distribution over
 * functions:
 *
 * \f[
 * f(x) \sim \mathcal{GP} (m(x), k(x,x'))
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

	/** returns name of the machine
	 *
	 * @return name GaussianProcessMachine
	 */
	virtual const char* get_name() const { return "GaussianProcessMachine"; }

	/** returns a mean \f$\mu\f$ of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$.
	 *
	 * @param data testing features
	 *
	 * @return posterior means
	 */
	SGVector<float64_t> get_posterior_means(CFeatures* data);

	/** returns a variance \f$\sigma^2\f$ of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$.
	 *
	 * @param data testing features
	 *
	 * @return posterior variances
	 */
	SGVector<float64_t> get_posterior_variances(CFeatures* data);

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

	/** Stores feature data of underlying model. After this method has been
	 * called, it is possible to change the machine's feature data and call
	 * apply(), which is then performed on the training feature data that is
	 * part of the machine's model.
	 */
	virtual void store_model_features() { }

private:
	void init();

protected:
	/** inference method */
	CInferenceMethod* m_method;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _GAUSSIANPROCESSMACHINE_H_ */
