/*
 * Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2017 Michele Mazzoni
 */

#ifndef _DOTITERATOR_H___
#define _DOTITERATOR_H___

#include <shogun/features/DotFeatures.h>

namespace shogun
{

	class DotIterator
	{
		class iterator;

		class feature_vector
		{
		public:
			/** constructor
			 *
			 * @param features pointer to DotFeatures
			 * @param idx position of the iterator
			 */
			feature_vector(std::shared_ptr<DotFeatures> features, index_t idx)
			    : m_features(features), m_idx(idx)
			{
			}

			/** copy constructor */
			feature_vector(const feature_vector& orig)
			    : m_features(orig.m_features), m_idx(orig.m_idx)
			{
			}

			/** dot product between the current feature vector and another
			 * vector
			 *
			 * @param v vector
			 * @return dot product result
			 */
			float64_t dot(const SGVector<float64_t>& v) const;

			/** multiply the current feature vector to a scalar and add it to a
			 * vector
			 *
			 * @param alpha scalar to multiply the feature vector to
			 * @param v accumulator vector
			 */
			void add(float64_t alpha, SGVector<float64_t>& v) const;

		protected:
			std::shared_ptr<DotFeatures> m_features;
			index_t m_idx;

			friend class iterator;
		};

		class iterator
		{
			using value_type = feature_vector;
			using difference_type = index_t;
			using pointer = feature_vector*;
			using reference = feature_vector&;
			using iterator_category = std::forward_iterator_tag;

		public:
			/** default constructor */
			iterator() : m_feature_vector(nullptr, -1)
			{
			}

			/** constructor
			 *
			 * @param features pointer to DotFeatures
			 * @param idx position of the iterator
			 */
			iterator(std::shared_ptr<DotFeatures> features, index_t idx = 0)
			    : m_feature_vector(features, idx)
			{
			}

			/** dereference operator */
			reference operator*();

			/** pre-increment operator */
			iterator& operator++();

			/** post-increment operator */
			iterator operator++(int);

			/** equality operator */
			bool operator==(const iterator& rhs);

			/** inequality operator */
			bool operator!=(const iterator& rhs);

		protected:
			value_type m_feature_vector;
		};

	public:
		/** constructor
		 *
		 * @param features pointer to DotFeatures
		 */
		DotIterator(std::shared_ptr<DotFeatures> features) : m_features(features)
		{
		}

		/** iterator pointing to the first feature vector
		 *
		 * @return iterator
		 */
		iterator begin() const;

		/** iterator pointing to the past-the-end feature vector
		 *
		 * @return iterator
		 */
		iterator end() const;

	protected:
		std::shared_ptr<DotFeatures> m_features;
	};
}

#endif // _DOTITERATOR_H___
