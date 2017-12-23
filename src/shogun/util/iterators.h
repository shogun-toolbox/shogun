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
 * Authors: 2017 Viktor Gal
 */

#ifndef SHOGUN_ITERATORS_H_
#define SHOGUN_ITERATORS_H_

namespace shogun
{
		template<typename T>
		class RandomIterator
		{
		public:
			// iterator traits
			using difference_type = std::ptrdiff_t;
			using value_type = T;
			using pointer = T*;
			using reference = T&;
			using iterator_category = std::random_access_iterator_tag;

			explicit RandomIterator(pointer ptr) : m_ptr(ptr) {}

			RandomIterator& operator++() { m_ptr++; return *this; }
			RandomIterator operator++(int) { RandomIterator retval = *this; m_ptr++; return retval;}
			RandomIterator& operator--() { m_ptr--; return *this; }
			RandomIterator operator--(int) { RandomIterator retval = *this; m_ptr--; return retval;}

			bool operator==(const RandomIterator& other) const { return m_ptr == other.m_ptr; }
			bool operator!=(const RandomIterator& other) const { return m_ptr != other.m_ptr; }

			reference operator*() {return *m_ptr;}
			pointer operator->() { return m_ptr; }

			RandomIterator &operator += (difference_type d) { m_ptr += d; return *this; }
			RandomIterator &operator -= (difference_type d) { m_ptr -= d; return *this; }

			RandomIterator operator + (difference_type d) const { return RandomIterator(m_ptr+d); }
			RandomIterator operator - (difference_type d) const { return RandomIterator(m_ptr-d); }

			reference operator [] (difference_type d) const { return m_ptr[d]; }

			bool operator < (const RandomIterator &other) const { return m_ptr < other.m_ptr; }
			bool operator > (const RandomIterator &other) const { return m_ptr > other.m_ptr; }
			bool operator <= (const RandomIterator &other) const { return m_ptr <= other.m_ptr; }
			bool operator >= (const RandomIterator &other) const { return m_ptr >= other.m_ptr; }

			difference_type operator - (const RandomIterator &other) const { return m_ptr - other.m_ptr; }

		private:
			pointer m_ptr;
		};
}

#endif
