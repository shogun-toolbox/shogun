/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Written (W) 2014 Dhruv Jawali (dhruv13.j@gmail.com)
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
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
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
		/** The main CDistance Object */
		CDist *dist;
	
		SGDataSetAdaptor(CDist *dist_) : dist(dist_) { }
		
		/** CRTP Helper Function
		  * @return CDistance Object
		  */
		inline CDist& distance() { return *dist; }
	
		/** Get the number of Training Set Data Points
		  * @return Number of features within CDistance::lhs
		  */
		inline size_t kdtree_get_point_count()
		{
			CDenseFeatures<ElementType> *f = ((CDenseFeatures<ElementType>*) distance().get_lhs());
			size_t num =  f->get_num_vectors();
			SG_UNREF(f);
			return num;
		}
		
		/** Get the dim'th component of the idx'th feature vector
		  * @param idx Index of the feature vector
		  * @param dim The component of the feature vector to fetch
		  * @return value
		  */
		inline ElementType kdtree_get_pt(int32_t idx, int dim)
		{
			ElementType val;
			
			CDenseFeatures<ElementType> *f = ((CDenseFeatures<ElementType>*) distance().get_lhs());
			SGVector<ElementType> vec = f->get_feature_vector(idx);
			val = vec[dim];
			f->free_feature_vector(vec, idx);
			
			SG_UNREF(f);
			return val;
		}
		
		/** Optional bounding box computation: false return value defaults to a 
		  * standard bbox computation loop.
		  */
		template <class BBOX>
			bool kdtree_get_bbox(BBOX &bb) { return false; }
};

#endif
