/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2014 Dhruv Jawali (dhruv13.j@gmail.com)
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef _SGADAPTOR_H__
#define _SGADAPTOR_H__

#include <shogun/lib/nanoflann/nanoflann.hpp>

/** Adaptor Class to use the data stored within CDistance objects directly to 
  * build and search on a kdtree. This is called within KNN.h to add kdtree mode
  * support.
  *
  *	@param CDist Type of CDistance Object (currently supports CEuclideanDistance)
  * @param ElementType Type of data held within CDistance::{lhs, rhs}
  */
template <typename CDist, class ElementType = float64_t>
class SGDataSetAdaptor
{
	public:
		const CDist *dist;
	
		SGDataSetAdaptor(const CDist *dist_) : dist(dist_) { }
		inline const CDist& distance() const { return *dist; }
	
		inline size_t kdtree_get_point_count() const
		{
			CDenseFeatures<ElementType> *f = ((CDenseFeatures<ElementType>*) distance().get_lhs());
			size_t num =  f->get_num_vectors();
			SG_UNREF(f);
			return num;
		}
	
		inline ElementType kdtree_get_pt(const int32_t idx, int dim) const
		{
			ElementType val;
			
			CDenseFeatures<ElementType> *f = ((CDenseFeatures<ElementType>*) distance().get_lhs());
			SGVector<ElementType> vec = f->get_feature_vector(idx);
			val = vec[dim];
			
			SG_UNREF(f);
			return val;
		}
	
		template <class BBOX>
			bool kdtree_get_bbox(BBOX &bb) const { return false; }
};

#endif
