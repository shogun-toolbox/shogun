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

#ifndef _KNNHEAP_H__
#define  _KNNHEAP_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief This class implements a specialized version of max heap structure. This heap specializes in storing the least
 * k values seen so far along with the indices (or id) of the entities with which the values are associated. On calling
 * the push method, it is automatically checked, if the new value supplied, is among the least k distances seen so far. Also,
 * in case the heap is full already, the max among the stored values is automatically thrown out as the new value finds its
 * proper place in the heap.
 */
class CKNNHeap
{
public:
	/** constructor
	 *
	 * @param k heap capacity i.e. the number of least distance values to be stored 
	 */
	CKNNHeap(int32_t k=1);

	/** destructor */
	~CKNNHeap() { };

	/** push into heap
	 *
	 * @param index vector id whose distance value is pushed into the heap
	 * @param dist distance value of the vector id index from the query point
	 */
	void push(index_t index, float64_t dist);

	/** max distance
	 *
	 * @return max distance value stored in the heap
	 */
	float64_t get_max_dist() { return m_dists[0]; }

	/** max index
	 *
	 * @return vector id of the max distance stored
	 */
	index_t get_max_index() { return m_inds[0]; }

	/** get distances
	 *
	 * @return distances stored in the heap
	 */
	SGVector<float64_t> get_dists();

	/** get indices
	 *
	 * @return indices stored in the heap
	 */
	SGVector<index_t> get_indices();

private:
	/** distance heap */
	SGVector<float64_t> m_dists;

	/** vector indices corresponding to distances */
	SGVector<index_t> m_inds;

	/** heap capacity */
	int32_t m_capacity;

	/** whether distances are already sorted for output */
	bool m_sorted;
};
}
#endif /* KNNHEAP */