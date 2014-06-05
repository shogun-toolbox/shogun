/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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


#ifndef _RANDOMCARTREE_H__
#define _RANDOMCARTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/CARTree.h>

namespace shogun
{
class CRandomCARTree : public CCARTree
{
public:
	/** constructor */
	CRandomCARTree();

	/** destructor */
	virtual ~CRandomCARTree();

	/** get name
	 * @return class name CARTree
	 */
	virtual const char* get_name() const { return "RandomCARTree"; }

	/** set number of random features to choose in each node split
	 *
	 * @param size subset size
	 */
	void set_feature_subset_size(int32_t size);

	/** get number of random features to choose in each node split
	 *
	 * @return size subset size
	 */
	int32_t get_feature_subset_size() const { return m_randsubset_size; }
protected:
	/** CARTtrain - recursive CART training method
	 *
	 * @param data training data
	 * @param weights vector of weights of data points
	 * @param labels labels of data points
	 * @return pointer to the root of the CART subtree
	 */
	virtual CBinaryTreeMachineNode<CARTreeNodeData>* CARTtrain(CFeatures* data, SGVector<float64_t> weights, CLabels* labels);

private:
	/** initialize parameters */
	void init();

private:
	/** random feature subset size */
	int32_t m_randsubset_size;

};
} /* namespace shogun */

#endif /* _RANDOMCARTREE_H__ */
