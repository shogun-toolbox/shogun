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

#include <shogun/multiclass/tree/BallTree.h>

using namespace shogun;

CBallTree::CBallTree(int32_t leaf_size, EDistanceMetric d)
: CNbodyTree(leaf_size,d)
{
}

float64_t CBallTree::min_distsq(bnode_t* node,float64_t* feat, int32_t dim)
{
	float64_t dist=0;
	SGVector<float64_t> center=node->data.center;
	for (int32_t i=0;i<dim;i++)
		dist+=add_dim_dist(center[i]-feat[i]);

	return CMath::max(0.0,dist-node->data.radius);
}

void CBallTree::init_node(bnode_t* node, index_t start, index_t end)
{
	SGVector<float64_t> upper_bounds(m_data.num_rows);
	SGVector<float64_t> lower_bounds(m_data.num_rows);

	SGVector<float64_t> center(m_data.num_rows);
	for (int32_t i=0;i<m_data.num_rows;i++)
	{
		center[i]=m_data(i,vec_id[start]);
		upper_bounds[i]=m_data(i,vec_id[start]);
		lower_bounds[i]=m_data(i,vec_id[start]);
		for (int32_t j=start+1;j<=end;j++)
		{
			float64_t data_pt=m_data(i,vec_id[j]);
			upper_bounds[i]=CMath::max(upper_bounds[i],data_pt);
			lower_bounds[i]=CMath::min(lower_bounds[i],data_pt);
			center[i]+=data_pt;
		}

		center[i]/=(end-start+1.f);
	}

	float64_t radius=0;
	for (int32_t i=start;i<=end;i++)
		radius=CMath::max(distance(vec_id[i],center.vector,center.vlen),radius);

	actual_dists(&radius,1);

	node->data.radius=radius;
	node->data.center=center;
	node->data.start_idx=start;
	node->data.bbox_upper=upper_bounds;
	node->data.bbox_lower=lower_bounds;	
	node->data.end_idx=end;
}