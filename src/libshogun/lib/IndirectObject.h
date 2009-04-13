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

#include "lib/common.h"

/** @brief an array class that accesses elements indirectly via an index array.
 *
 * It does not store the objects itself, but only indices to objects.
 * This conveniently allows e.g. sorting the array without changing
 * the order of objects (but only the order of their indices).
 */
template <class T> class CIndirectObject
{
	public:
		CIndirectObject() : index(-1)
		{
		}

		CIndirectObject(int32_t idx)
		{
			index=idx;
		}

		static void set_array(T* a)
		{
			array=a;
		}

		static void init_slice(CIndirectObject<T>* a, int32_t len)
		{
			for (int32_t i=0; i<len; i++)
				a[i].index=i;
		}

		/** overload = operator
		 * @param x assign elements from x
		 */
		CIndirectObject<T>& operator=(const CIndirectObject<T>& x)
		{ 
			index=x.index;
			return *this; 
		}

		/** overload | operator and return x | y 
		 *
		 * @param x x
		 */
		T operator|(const CIndirectObject<T>& x) const
		{
			return array[index] | x.array[x.index];
		}

		/** overload & operator and return x & y 
		 *
		 * @param x x
		 */
		const T operator&(const CIndirectObject<T>& x) const
		{
			return array[index] & x.array[x.index];
		}

		/** overload << operator
		 *
		 * perform bit shift to the left
		 *
		 * @param shift shift by this amount
		 */
		T operator<<(int shift)
		{
			return array[index] << shift;
		}

		/** overload >> operator
		 *
		 * perform bit shift to the right
		 *
		 * @param shift shift by this amount
		 */
		T operator>>(int shift)
		{
			return array[index] >> shift;
		}

		/** overload ^ operator and return x ^ y 
		 *
		 * @param x x
		 */
		T operator^(const CIndirectObject<T>& x) const
		{
			return array[index] ^ x.array[x.index];
		}

		/** overload + operator and return x + y 
		 *
		 * @param x x
		 */
		T operator+(const CIndirectObject<T> &x) const
		{
			return array[index] + x.array[x.index];
		}

		/** overload - operator and return x - y 
		 *
		 * @param x x
		 */
		T operator-(const CIndirectObject<T> &x) const
		{
			return array[index] - x.array[x.index];
		}

		/** overload / operator and return x / y 
		 *
		 * @param x x
		 */
		T operator/(const CIndirectObject<T> &x) const
		{
			return array[index] / x.array[x.index];
		}

		/** overload * operator and return x * y 
		 *
		 * @param x x
		 */
		T operator*(const CIndirectObject<T> &x) const
		{
			return array[index] * x.array[x.index];
		}

		/** overload += operator; add x to current element
		 *
		 * @param x x
		 */
		CIndirectObject<T>& operator+=(const CIndirectObject<T> &x)
		{
			array[index]+=x.array[x.index];
			return *this;
		}

		/** overload -= operator; substract x from current element
		 *
		 * @param x x
		 */
		CIndirectObject<T>& operator-=(const CIndirectObject<T> &x)
		{
			array[index]-=x.array[x.index];
			return *this;
		}

		/** overload *= operator; multiple x to with current element
		 *
		 * @param x x
		 */
		CIndirectObject<T>& operator*=(const CIndirectObject<T> &x)
		{
			array[index]*=x.array[x.index];
			return *this;
		}

		/** overload /= operator; divide current object by x
		 *
		 * @param x x
		 */
		CIndirectObject<T>& operator/=(const CIndirectObject<T> &x)
		{
			array[index]/=x.array[x.index];
			return *this;
		}

		/** overload == operator; test if current object equals x
		 *
		 * @param x x
		 */
		bool operator==(const CIndirectObject<T> &x) const
		{
			return array[index]==x.array[x.index];
		}

		/** overload >= operator; test if current object greater equal x
		 *
		 * @param x x
		 */
		bool operator>=(const CIndirectObject<T> &x) const
		{
			return array[index]>=x.array[x.index];
		}

		/** overload <= operator; test if current object lower equal x
		 *
		 * @param x x
		 */
		bool operator<=(const CIndirectObject<T> &x) const
		{
			return array[index]<=x.array[x.index];
		}

		/** overload > operator; test if current object is bigger than x
		 *
		 * @param x x
		 */
		bool operator>(const CIndirectObject<T> &x) const
		{
			return array[index]>x.array[x.index];
		}

		/** overload < operator; test if current object is smaller than x
		 *
		 * @param x x
		 */
		bool operator<(const CIndirectObject<T> &x) const
		{
			return array[index]<x.array[x.index];
		}

		/** overload ! operator; test if current object is not equal to x
		 *
		 * @param x x
		 */
		bool operator!=(const CIndirectObject<T> &x) const
		{
			return array[index]!=x.array[x.index];
		}

		/** overload |= operator
		 *
		 * perform bitwise or with current element and x
		 *
		 * @param x x
		 */
		CIndirectObject<T>& operator|=(const CIndirectObject<T>& x)
		{
			array[index]|=x.array[x.index];
			return *this;
		}

		/** overload &= operator
		 *
		 * perform bitwise and with current element and x
		 *
		 * @param x x
		 */
		CIndirectObject<T>& operator&=(const CIndirectObject<T>& x)
		{
			array[index]&=x.array[x.index];
			return *this;
		}

		/** overload ^= operator
		 *
		 * perform bitwise xor with current element and x
		 *
		 * @param x x
		 */
		CIndirectObject<T>& operator^=(const CIndirectObject<T>& x)
		{
			array[index]^=x.array[x.index];
			return *this;
		}

		/** overload <<= operator
		 *
		 * perform bit shift to the left
		 *
		 * @param shift shift by this amount
		 */
		CIndirectObject<T>& operator<<=(int shift)
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
		CIndirectObject<T>& operator>>=(int shift)
		{
			*this=*this>>shift;
			return *this;
		}

		/** negate element */
		T operator~()
		{
			return ~array[index];
		}

		/** return array element */
		operator T() const { return array[index]; }

		/** decrement element by one */
		CIndirectObject<T>& operator--()
		{
			array[index]--;
			return *this;
		}

		/** increment element by one */
		CIndirectObject<T>& operator++()
		{
			array[index]++;
			return *this;
		}

	protected:
		/** array */
		static T* array;

		/** index into array */
		int32_t index;
};
#endif //__INDIRECTOBJECT_H__
