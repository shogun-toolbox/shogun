/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2016 Saurabh Mahindre
 */

#ifndef _RRCARTREE_H__
#define _RRCARTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/RandomCARTree.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief This class implements the Random Rotation based Classification and Regression Trees.
 *  
 *  For more information refer: Random Rotation Ensembles Journal of Machine Learning Research, Vol. 17, No. 4. (2016), pp. 1-26 by Rico Blaser, Piotr Fryzlewicz
 */	
class CRRCARTree : public CRandomCARTree
{
public:
	
	/** constructor */
	CRRCARTree();

	/** destructor */
	virtual ~CRRCARTree();

	/** get name
	 * @return class name CARTree
	 */
	virtual const char* get_name() const { return "RRCARTree"; }

	/** classify data using Classification Tree
	 * @param data data to be classified
	 * @return MulticlassLabels corresponding to labels of various test vectors
	 */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

	/** Get regression labels using Regression Tree
	 * @param data data whose regression output is needed
	 * @return Regression output for various test vectors
	 */
	virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

protected:
	/** Generate n x n rotation matrix
	  * @param n matrix dimension
	  */
	virtual SGMatrix<float64_t> generate_rotation_matrix(int32_t n);

	virtual bool train_machine(CFeatures* data = NULL);

protected:
	/* Generated Random rotation matrix */
	SGMatrix<float64_t> m_rotation_matrix;
};
}
#endif

