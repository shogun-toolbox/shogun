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

#include <shogun/multiclass/tree/KNNHeap.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h> 

using namespace shogun;

CKNNHeap::CKNNHeap(int32_t k)
{
	capacity=k;
	dists=SGVector<float64_t>(capacity);
	inds=SGVector<index_t>(capacity);
	sorted=false;

	for (int32_t i=0;i<capacity;i++)
	{
		dists[i]=CMath::MAX_REAL_NUMBER;
		inds[i]=0;
	}
}

void CKNNHeap::push(index_t index, float64_t dist)
{
	if (dist>dists[0])
		return;

	dists[0]=dist;
	inds[0]=index;

	index_t i_swap;
	index_t i=0;
	while (true)
	{
		index_t l=2*i+1;
		index_t r=l+1;
		if (l>=capacity)
		{
			break;
		}
		else if (r>=capacity)
		{
			if (dists[l]>dist)
				i_swap=l;
			else
				break;
		}
		else if (dists[l]>=dists[r])
		{
			if (dists[l]>dist)
				i_swap=l;
			else
				break;
		}
		else
		{
			if (dists[r]>dist)
				i_swap=r;
			else
				break;
		}

		dists[i]=dists[i_swap];
		inds[i]=inds[i_swap];

		dists[i_swap]=dist;
		inds[i_swap]=index;
		i=i_swap;
	}
}

SGVector<float64_t> CKNNHeap::get_dists()
{
	if (sorted)
		return dists;

	sorted=true;
	SGVector<float64_t> new_dists(capacity);
	SGVector<index_t> new_inds(capacity);

	// O(nlogn) heap-sort
	for (int32_t i=capacity-1;i>-1;i--)
	{
		new_dists[i]=dists[0];
		new_inds[i]=inds[0];
		push(0,-1);
	}

	dists=new_dists;
	inds=new_inds;

	return dists;
}

SGVector<index_t> CKNNHeap::get_indices()
{
	if (sorted)
		return inds;

	sorted=true;
	SGVector<float64_t> new_dists(capacity);
	SGVector<index_t> new_inds(capacity);

	// O(nlogn) heap-sort
	for (int32_t i=capacity-1;i>-1;i--)
	{
		new_dists[i]=dists[0];
		new_inds[i]=inds[0];
		push(0,-1);
	}

	dists=new_dists;
	inds=new_inds;

	return inds;
}
