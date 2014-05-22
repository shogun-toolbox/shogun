/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 */

#ifndef _VARIATIONALLIKELIHOODMODEL_H_
#define _VARIATIONALLIKELIHOODMODEL_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/gp/LikelihoodModel.h>

namespace shogun
{

template<class T> class SGVector;

/** @brief The Variational Likelihood base class.
 *
 */
class CVariationalLikelihood : public CLikelihoodModel
{
public:
	/** default constructor */
	CVariationalLikelihood() : CLikelihoodModel() {};

	virtual ~CVariationalLikelihood() {};

	/** get model type
	  *
	  * @return model type NONE
	 */
	virtual ELikelihoodModelType get_model_type() const { return LT_NONE; }

	/** returns the expection of the logarithm of a given probability distribution 
	 * wrt the variational distribution.
	 *
	 * @return expection
	 */
	virtual SGVector<float64_t> get_variational_expection() =0;

	/** get derivative of the variational expection of log likelihood 
	 * with respect to given parameter
	 *
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_variational_first_derivative(const TParameter* param) const =0;

};
}
#endif /* _VARIATIONALLIKELIHOODMODEL_H_ */
