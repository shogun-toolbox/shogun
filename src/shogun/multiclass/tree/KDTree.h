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


#ifndef _KDTREE_H__
#define _KDTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/NbodyTree.h>

namespace shogun
{

/** @brief This class implements KD-Tree.
 */
class CKDTree : public CNbodyTree
{
public:
	/** constructor
	 * 
	 * @param data data points using which KD-Tree will be made	 
	 * @param leaf_size min number of samples in any node
	 */
	CKDTree(int32_t leaf_size=1, EDistanceMetric d=DM_EUCLID);
	
	/** Destructor */
	~CKDTree();

	/** get name
	 * @return class of the tree 
	 */
	virtual const char* get_name() const { return "KDTree"; }

private:
	/** find squared minimum distance between node and a query vector
	 * 
	 * @param node present node
	 * @param feat query vector
	 * @param dim dimensions of query vector
	 * @return squared min distance
	 */
	float64_t min_distsq(bnode_t* node,float64_t* feat, int32_t dim);

	/** initialize node
	 *
	 * @param node node to be initialized
	 * @param start start index of index vector
	 * @param end end index of index vector
	 */
	void init_node(bnode_t* node, index_t start, index_t end);

};
} /* namespace shogun */

#endif /* _KDREE_H__ */
