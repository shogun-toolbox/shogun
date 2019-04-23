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

#include <shogun/features/iterators/DotIterator.h>

namespace shogun
{

	float64_t
	DotIterator::feature_vector::dot(const SGVector<float64_t>& v) const
	{
		return m_features->dense_dot(m_idx, v.vector, v.vlen);
	}

	void DotIterator::feature_vector::add(
	    float64_t alpha, SGVector<float64_t>& v) const
	{
		m_features->add_to_dense_vec(alpha, m_idx, v.vector, v.vlen);
	}

	typename DotIterator::iterator::reference DotIterator::iterator::operator*()
	{
		return m_feature_vector;
	}

	typename DotIterator::iterator& DotIterator::iterator::operator++()
	{
		++(m_feature_vector.m_idx);
		return *this;
	}

	typename DotIterator::iterator DotIterator::iterator::operator++(int)
	{
		iterator tmp(*this);
		++(*this);
		return tmp;
	}

	bool DotIterator::iterator::operator==(const iterator& rhs)
	{
		return m_feature_vector.m_idx == rhs.m_feature_vector.m_idx;
	}

	bool DotIterator::iterator::operator!=(const iterator& rhs)
	{
		return !(*this == rhs);
	}

	DotIterator::iterator DotIterator::begin() const
	{
		return iterator(m_features, 0);
	}

	DotIterator::iterator DotIterator::end() const
	{
		return iterator(m_features, m_features->get_num_vectors());
	}
}
