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

/** @brief This class implements randomized CART algorithm used in the tree growing process of candidate trees in Random Forests algorithm.
 * The tree growing process is different from the original CART algorithm because of the input attributes which are considered for each node  
 * split. In randomized CART, a few (fixed number) attributes are randomly chosen from all available attributes while deciding the best split.
 * This is unlike the original CART where all available attributes are considered while deciding the best split.
 */
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
	/** computes best attribute for CARTtrain
	 *
	 * @param mat data matrix
	 * @param weights data weights
	 * @param labels_vec data labels
	 * @param left stores feature values for left transition
	 * @param right stores feature values for right transition
	 * @param is_left_final stores which feature vectors go to the left child
	 * @param num_missing number of missing attributes
	 * @param count_left stores number of feature values for left transition
	 * @param count_right stores number of feature values for right transition
	 * @return index to the best attribute
	 */
	virtual int32_t compute_best_attribute(SGMatrix<float64_t> mat, SGVector<float64_t> weights, SGVector<float64_t> labels_vec, 	
	SGVector<float64_t> left, SGVector<float64_t> right, SGVector<bool> is_left_final, int32_t &num_missing, int32_t &count_left,
														 int32_t &count_right);

private:
	/** initialize parameters */
	void init();

private:
	/** random feature subset size */
	int32_t m_randsubset_size;

};
} /* namespace shogun */

#endif /* _RANDOMCARTREE_H__ */
