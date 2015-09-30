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
	m_capacity=k;
	m_dists=SGVector<float64_t>(m_capacity);
	m_inds=SGVector<index_t>(m_capacity);
	m_sorted=false;

	for (int32_t i=0;i<m_capacity;i++)
	{
		m_dists[i]=CMath::MAX_REAL_NUMBER;
		m_inds[i]=0;
	}
}

void CKNNHeap::push(index_t index, float64_t dist)
{
	if (dist>m_dists[0])
		return;

	m_dists[0]=dist;
	m_inds[0]=index;

	index_t i_swap;
	index_t i=0;
	while (true)
	{
		index_t l=2*i+1;
		index_t r=l+1;
		if (l>=m_capacity)
		{
			break;
		}
		else if (r>=m_capacity)
		{
			if (m_dists[l]>dist)
				i_swap=l;
			else
				break;
		}
		else if (m_dists[l]>=m_dists[r])
		{
			if (m_dists[l]>dist)
				i_swap=l;
			else
				break;
		}
		else
		{
			if (m_dists[r]>dist)
				i_swap=r;
			else
				break;
		}

		m_dists[i]=m_dists[i_swap];
		m_inds[i]=m_inds[i_swap];

		m_dists[i_swap]=dist;
		m_inds[i_swap]=index;
		i=i_swap;
	}
}

SGVector<float64_t> CKNNHeap::get_dists()
{
	if (m_sorted)
		return m_dists;

	m_sorted=true;
	SGVector<float64_t> new_dists(m_capacity);
	SGVector<index_t> new_inds(m_capacity);

	// O(nlogn) heap-sort
	for (int32_t i=m_capacity-1;i>-1;i--)
	{
		new_dists[i]=m_dists[0];
		new_inds[i]=m_inds[0];
		push(0,-1);
	}

	m_dists=new_dists;
	m_inds=new_inds;

	return m_dists;
}

SGVector<index_t> CKNNHeap::get_indices()
{
	if (m_sorted)
		return m_inds;

	m_sorted=true;
	SGVector<float64_t> new_dists(m_capacity);
	SGVector<index_t> new_inds(m_capacity);

	// O(nlogn) heap-sort
	for (int32_t i=m_capacity-1;i>-1;i--)
	{
		new_dists[i]=m_dists[0];
		new_inds[i]=m_inds[0];
		push(0,-1);
	}

	m_dists=new_dists;
	m_inds=new_inds;

	return m_inds;
}
