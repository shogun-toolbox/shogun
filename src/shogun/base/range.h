/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2016 Sergey Lisitsyn
*
*/

#ifndef __SG_RANGE_H__
#define __SG_RANGE_H__

#include <iterator>

namespace shogun
{

	/** @class Helper class to spawn range iterator.
	 *
	 * Useful for C++11-style for loops:
	 *
	 * @code
	 *  for (auto i : Range(3, 10)) { ... }
	 * @endcode
	 */
	template <typename T>
	class Range
	{
	public:
		/** Creates range with specified bounds.
		 * Assumes rbegin < rend.
		 *
		 * @param   rbegin   lower bound of range
		 * @param   rend     upper bound of range (excluding)
		 */
		Range(T rbegin, T rend) : m_begin(rbegin), m_end(rend)
		{
		}

		/** @class Iterator spawned by @ref Range. */
		class Iterator : public std::iterator<std::input_iterator_tag, T>
		{
		public:
			Iterator(T value) : m_value(value)
			{
			}
			Iterator(const Iterator& other) : m_value(other.m_value)
			{
			}
			Iterator(Iterator&& other) : m_value(other.m_value)
			{
			}
			Iterator& operator=(const Iterator&) = delete;
			Iterator& operator++()
			{
				m_value++;
				return *this;
			}
			Iterator operator++(int)
			{
				Iterator tmp(*this);
				++*this;
				return tmp;
			}
			T operator*()
			{
				return m_value;
			}
			bool operator!=(const Iterator& other)
			{
				return this->m_value != other.m_value;
			}

		private:
			T m_value;
		};
		/** Create iterator that corresponds to the start of range.
		 *
		 * Usually called through for-loop syntax.
		 */
		Iterator begin() const
		{
			return Iterator(m_begin);
		}
		/** Create iterator that corresponds to the end of range.
		 *
		 * Usually called through for-loop syntax.
		 */
		Iterator end() const
		{
			return Iterator(m_end);
		}

	private:
		/** begin of range */
		T m_begin;
		/** end of range */
		T m_end;
	};

	/** Creates @ref Range with specified upper bound.
	 *
	 * @code
	 *  for (auto i : range(100)) { ... }
	 * @endcode
	 *
	 * @param   rend     upper bound of range (excluding)
	 */
	template <typename T>
	inline Range<T> range(T rend)
	{
		return Range<T>(0, rend);
	}

	/** Creates @ref Range with specified bounds.
	 *
	 * @code
	 *  for (auto i : range(0, 100)) { ... }
	 * @endcode
	 *
	 * @param   rbegin  lower bound of range
	 * @param   rend    upper bound of range (excluding)
	 */
	template <typename T>
	inline Range<T> range(T rbegin, T rend)
	{
		return Range<T>(rbegin, rend);
	}
}

#endif /* __SG_RANGE_H__ */
