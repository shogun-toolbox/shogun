/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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
 */

#ifndef BAHSIC_H__
#define BAHSIC_H__

#include <shogun/lib/config.h>
#include <shogun/preprocessor/KernelDependenceMaximization.h>

namespace shogun
{

/** @brief Class CBAHSIC, that extends CKernelDependenceMaximization and uses
 * HSIC [1] to compute dependence measures for feature selection using a
 * backward elimination approach as described in [1]. This class serves as a
 * convenience class that initializes the CDependenceMaximization#m_estimator
 * with an instance of CHSIC and allows only shogun::BACKWARD_ELIMINATION algorithm
 * to use which is set internally. Therefore, trying to use other algorithms
 * by set_algorithm() will not work. Plese see the class documentation of CHSIC
 * and [2] for more details on mathematical description of HSIC.
 *
 * Refrences:
 * [1] Song, Le and Bedo, Justin and Borgwardt, Karsten M. and Gretton, Arthur
 * and Smola, Alex. (2007). Gene Selection via the BAHSIC Family of Algorithms.
 * Journal Bioinformatics. Volume 23 Issue Pages i490-i498. Oxford University
 * Press Oxford, UK
 * [2]: Gretton, A., Fukumizu, K., Teo, C., & Song, L. (2008). A kernel
 * statistical test of independence. Advances in Neural Information Processing
 * Systems, 1-8.
 */
class CBAHSIC : public CKernelDependenceMaximization
{
public:
	/** Default constructor */
	CBAHSIC();

	/** Destructor */
	virtual ~CBAHSIC();

	/**
	 * Since only shogun::BACKWARD_ELIMINATION algorithm is applicable for BAHSIC,
	 * and this is set internally, this method is overridden to prevent this
	 * to be set from public API.
	 *
	 * @param algorithm the feature selection algorithm to use
	 */
	virtual void set_algorithm(EFeatureSelectionAlgorithm algorithm);

	/** @return the preprocessor type */
	virtual EPreprocessorType get_type() const;

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "BAHSIC";
	}

private:
	/** Register params and initialize with default values */
	void initialize_preprocessor();

};

}
#endif // BAHSIC_H__
