/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max Planck Society
 */
#ifndef __INDIRECTOBJECT_H__
#define __INDIRECTOBJECT_H__

#include <lib/common.h>

namespace shogun
{
/** @brief an array class that accesses elements indirectly via an index array.
 *
 * It does not store the objects itself, but only indices to objects.
 * This conveniently allows e.g. sorting the array without changing
 * the order of objects (but only the order of their indices).
 */
template <class T, class P> class CIndirectObject
{
	public:
		/** default constructor
		 * (initializes index with -1)
		 */
		CIndirectObject() : index(-1)
		{
		}

		/** constructor
		 * @param idx index
		 */
		CIndirectObject(int32_t idx)
		{
			index=idx;
		}

		/** set array
		 *
		 * @param a array
		 */
		static void set_array(P a)
		{
			array=a;
		}

		/** get array
		 *
		 * @return array
		 */
		static P get_array()
		{
			return array;
		}

		/** initialize slice
		 *
		 * @return array
		 */
		static void init_slice(CIndirectObject<T,P>* a, int32_t len, int32_t start=0, int32_t stop=-1)
		{
			if (stop==-1)
				stop=len;

			for (int32_t i=start; i<stop && i<len; i++)
				a[i].index=i;
		}

		/** overload = operator
		 * @param x assign elements from x
		 */
		CIndirectObject<T,P>& operator=(const CIndirectObject<T,P>& x)
		{
			index=x.index;
			return *this;
		}

		/** overload | operator and return x | y
		 *
		 * @param x x
		 */
		T operator|(const CIndirectObject<T,P>& x) const
		{
			return (*array)[index] | *(x.array)[x.index];
		}

		/** overload & operator and return x & y
		 *
		 * @param x x
		 */
		const T operator&(const CIndirectObject<T,P>& x) const
		{
			return (*array)[index] & *(x.array)[x.index];
		}

		/** overload << operator
		 *
		 * perform bit shift to the left
		 *
		 * @param shift shift by this amount
		 */
		T operator<<(int shift)
		{
			return (*array)[index] << shift;
		}

		/** overload >> operator
		 *
		 * perform bit shift to the right
		 *
		 * @param shift shift by this amount
		 */
		T operator>>(int shift)
		{
			return (*array)[index] >> shift;
		}

		/** overload ^ operator and return x ^ y
		 *
		 * @param x x
		 */
		T operator^(const CIndirectObject<T,P>& x) const
		{
			return (*array)[index] ^ *(x.array)[x.index];
		}

		/** overload + operator and return x + y
		 *
		 * @param x x
		 */
		T operator+(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index] + *(x.array)[x.index];
		}

		/** overload - operator and return x - y
		 *
		 * @param x x
		 */
		T operator-(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index] - *(x.array)[x.index];
		}

		/** overload / operator and return x / y
		 *
		 * @param x x
		 */
		T operator/(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index] / *(x.array)[x.index];
		}

		/** overload * operator and return x * y
		 *
		 * @param x x
		 */
		T operator*(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index] * *(x.array)[x.index];
		}

		/** overload += operator; add x to current element
		 *
		 * @param x x
		 */
		CIndirectObject<T,P>& operator+=(const CIndirectObject<T,P> &x)
		{
			(*array)[index]+=*(x.array)[x.index];
			return *this;
		}

		/** overload -= operator; substract x from current element
		 *
		 * @param x x
		 */
		CIndirectObject<T,P>& operator-=(const CIndirectObject<T,P> &x)
		{
			(*array)[index]-=*(x.array)[x.index];
			return *this;
		}

		/** overload *= operator; multiple x to with current element
		 *
		 * @param x x
		 */
		CIndirectObject<T,P>& operator*=(const CIndirectObject<T,P> &x)
		{
			(*array)[index]*=*(x.array)[x.index];
			return *this;
		}

		/** overload /= operator; divide current object by x
		 *
		 * @param x x
		 */
		CIndirectObject<T,P>& operator/=(const CIndirectObject<T,P> &x)
		{
			(*array)[index]/=*(x.array)[x.index];
			return *this;
		}

		/** overload == operator; test if current object equals x
		 *
		 * @param x x
		 */
		bool operator==(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index]==*(x.array)[x.index];
		}

		/** overload >= operator; test if current object greater equal x
		 *
		 * @param x x
		 */
		bool operator>=(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index]>=*(x.array)[x.index];
		}

		/** overload <= operator; test if current object lower equal x
		 *
		 * @param x x
		 */
		bool operator<=(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index]<=*(x.array)[x.index];
		}

		/** overload > operator; test if current object is bigger than x
		 *
		 * @param x x
		 */
		bool operator>(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index]>(*(x.array))[x.index];
		}

		/** overload < operator; test if current object is smaller than x
		 *
		 * @param x x
		 */
		bool operator<(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index]<(*(x.array))[x.index];
		}

		/** overload ! operator; test if current object is not equal to x
		 *
		 * @param x x
		 */
		bool operator!=(const CIndirectObject<T,P> &x) const
		{
			return (*array)[index]!=(*(x.array))[x.index];
		}

		/** overload |= operator
		 *
		 * perform bitwise or with current element and x
		 *
		 * @param x x
		 */
		CIndirectObject<T,P>& operator|=(const CIndirectObject<T,P>& x)
		{
			(*array)[index]|=(*(x.array))[x.index];
			return *this;
		}

		/** overload &= operator
		 *
		 * perform bitwise and with current element and x
		 *
		 * @param x x
		 */
		CIndirectObject<T,P>& operator&=(const CIndirectObject<T,P>& x)
		{
			(*array)[index]&=(*(x.array))[x.index];
			return *this;
		}

		/** overload ^= operator
		 *
		 * perform bitwise xor with current element and x
		 *
		 * @param x x
		 */
		CIndirectObject<T,P>& operator^=(const CIndirectObject<T,P>& x)
		{
			(*array)[index]^=(*(x.array))[x.index];
			return *this;
		}

		/** overload <<= operator
		 *
		 * perform bit shift to the left
		 *
		 * @param shift shift by this amount
		 */
		CIndirectObject<T,P>& operator<<=(int shift)
		{
			*this=*this<<shift;
			return *this;
		}

		/** overload >>= operator
		 *
		 * perform bit shift to the right
		 *
		 * @param shift shift by this amount
		 */
		CIndirectObject<T,P>& operator>>=(int shift)
		{
			*this=*this>>shift;
			return *this;
		}

		/** negate element */
		T operator~()
		{
			return ~(*array)[index];
		}

		/** return array element */
		operator T() const { return (*array)[index]; }

		/** decrement element by one */
		CIndirectObject<T,P>& operator--()
		{
			(*array)[index]--;
			return *this;
		}

		/** increment element by one */
		CIndirectObject<T,P>& operator++()
		{
			(*array)[index]++;
			return *this;
		}

	protected:
		/** array */
		static P array;

		/** index into array */
		int32_t index;
};

template <class T, class P> P CIndirectObject<T,P>::array;
}
#endif //__INDIRECTOBJECT_H__
