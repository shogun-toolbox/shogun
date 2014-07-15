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


#ifndef _BALLTREE_H__
#define _BALLTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/NbodyTree.h>

namespace shogun
{

/** @brief This class implements Ball tree. The ball tree is contructed using the top-down approach.
 * cf. ftp://ftp.icsi.berkeley.edu/pub/techreports/1989/tr-89-063.pdf
 */
class CBallTree : public CNbodyTree
{
public:
	/** constructor
	 * 
	 * @param data data points using which Ball-Tree will be made	 
	 * @param leaf_size min number of samples in any node
	 */
	CBallTree(int32_t leaf_size=1, EDistanceType d=D_EUCLIDEAN);
	
	/** Destructor */
	virtual ~CBallTree() { };

	/** get name
	 * @return class of the tree 
	 */
	virtual const char* get_name() const { return "BallTree"; }

private:
	/** find minimum distance between node and a query vector
	 * 
	 * @param node present node
	 * @param feat query vector
	 * @param dim dimensions of query vector
	 * @return min distance
	 */
	float64_t min_dist(bnode_t* node,float64_t* feat, int32_t dim);

	/** get min as well as max distance of a node from a point
	 *
	 * @param pt point whose distance is to be calculated
	 * @param node node from which distances are to be calculated
	 * @param lower lower bound of distance
	 * @param upper upper bound of distance
	 * @param dim dimension of point vector
	 */
	void min_max_dist(float64_t* pt, bnode_t* node, float64_t &lower,float64_t &upper, int32_t dim);

	/** initialize node
	 *
	 * @param node node to be initialized
	 * @param start start index of index vector
	 * @param end end index of index vector
	 */
	void init_node(bnode_t* node, index_t start, index_t end);

};
} /* namespace shogun */

#endif /* _BALLTREE_H__ */