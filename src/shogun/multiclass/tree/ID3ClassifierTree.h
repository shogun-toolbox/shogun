/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2013 Monica Dragan
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


#ifndef _ID3CLASSIFIERTREE_H__
#define _ID3CLASSIFIERTREE_H__

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/ID3TreeNodeData.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>

namespace shogun{

class CID3ClassifierTree : public CTreeMachine<id3TreeNodeData>
{
public:
	/** constructor */
	CID3ClassifierTree();

	/** destructor */
	virtual ~CID3ClassifierTree();

	/** get name
	 * @return class name ID3ClassifierTree
	 */
	virtual const char* get_name() const { return "ID3ClassifierTree"; }

	/** classify data using ID3 Tree
	 * @param data data to be classified 
	 */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

protected:
	
	/** train machine - build ID3 Tree from training data
	 * @param data training data
	 */
	virtual bool train_machine(CFeatures* data=NULL);

private:

	/** id3train - recursive id3 training method
	 *
	 * @param data training data
	 * @return pointer to the root of the ID3 tree
	 */
	node_t* id3train(CFeatures* data, CMulticlassLabels* 
					class_labels, SGVector<int32_t> values, int level = 0);
	
	/** informational gain attribute for selecting best feature at each node of ID3 Tree
	 *
	 * @param attr_no index to the chosen feature in data matrix supplied
	 * @param data data matrix
	 * @param class_labels classes to which corresponding data vectors belong
	 * @return informational gain of the chosen feature
	 */	
	float64_t informational_gain_attribute(int32_t attr_no, CFeatures* data, 
							CMulticlassLabels* class_labels);	
	
	/** computes entropy (aka randomness) in data
	 *
	 * @param labels lables of parameters chosen
	 * @return entropy
	 */		
	float64_t entropy(CMulticlassLabels* labels, float64_t*
			feature_values=NULL, float64_t active_value=0);
	
};
} /* namespace shogun */

#endif /* _ID3CLASSIFIERTREE_H__ */
