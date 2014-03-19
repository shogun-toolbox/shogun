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


#ifndef _ID3CLASSIFIERTREE_H__
#define _ID3CLASSIFIERTREE_H__

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/ID3TreeNodeData.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>

namespace shogun
{

/** @brief class ID3ClassifierTree, implements classifier tree for discrete feature 
 * values using the ID3 algorithm. The training algorithm implemented is as follows :
 *
 * function ID3 (R: a set of non-categorical attributes,
 *
 *		 C: the categorical attribute,
 *
 *		 S: a training set) returns a decision tree;
 *
 * begin
 *
 *	If S consists of records all with the same value for 
 *
 *	   the categorical attribute, 
 *
 *	   return a single node with that value;
 *
 *	If R is empty, then return a single node with as value
 *
 *	   the most frequent of the values of the categorical attribute
 *
 *	   that are found in records of S; [note that then there
 *
 *	   will be errors, that is, records that will be improperly
 *
 *	   classified];
 *
 *	Let D be the attribute with largest Gain(D,S) 
 *
 *	   among attributes in R;
 *
 *	Let \f${d_j| j=1,2, .., m}\f$ be the values of attribute D;
 *
 *	Let \f${S_j| j=1,2, .., m}\f$ be the subsets of S consisting 
 *
 *	   respectively of records with value dj for attribute D;
 *
 *	Return a tree with root labeled D and arcs labeled 
 *
 *	  \f$d_1, d_2, .., d_m\f$ going respectively to the trees 
 *
 *	ID3(R-{D}, C, \f$S_1\f$), .., ID3(R-{D}, C, \f$S_m\f$);
 *
 * end ID3;
 *
 */
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
	node_t* id3train(CFeatures* data, CMulticlassLabels* class_labels, SGVector<int32_t> values, int level = 0);
	
	/** informational gain attribute for selecting best feature at each node of ID3 Tree
	 *
	 * @param attr_no index to the chosen feature in data matrix supplied
	 * @param data data matrix
	 * @param class_labels classes to which corresponding data vectors belong
	 * @return informational gain of the chosen feature
	 */	
	float64_t informational_gain_attribute(int32_t attr_no, CFeatures* data, CMulticlassLabels* class_labels);	
	
	/** computes entropy (aka randomness) in data
	 *
	 * @param labels labels of parameters chosen
	 * @return entropy
	 */		
	float64_t entropy(CMulticlassLabels* labels);
	
};
} /* namespace shogun */

#endif /* _ID3CLASSIFIERTREE_H__ */
